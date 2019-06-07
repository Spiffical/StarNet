import os, sys
sys.path.insert(0, os.path.join(os.getenv('HOME'), 'StarNet'))
import time
import json
import datetime
import keras
import numpy as np
import tensorflow as tf
import multiprocessing
import h5py
import csv

from starnet.utils.data_utils.generator import DataGenerator
from starnet.utils.nn_utils.custom_callbacks import CustomModelCheckpoint, CustomReduceLROnPlateau
from starnet.utils.data_utils.loading import load_batch_from_h5, load_contiguous_slice_from_h5
from starnet.utils.data_utils.augment import add_zeros, telluric_mask
from keras.callbacks import EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model, Model
from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Input
from keras import Model
from keras.utils import multi_gpu_model

from sklearn.model_selection import train_test_split

get_session = keras.backend.get_session


def folder_runnum():
    """
    NAME:
        folder_runnum
    PURPOSE:
        to get the smallest available folder name without replacing the existing folder
    INPUT:
        None
    OUTPUT:
        folder name (string)
    HISTORY:
        2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    d = datetime.datetime.now()
    folder_name = None
    for runnum in range(1, 99999):
        folder_name = 'StarNet_{:%Y-%m-%d}_run{:d}'.format(d, runnum)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            break
        else:
            runnum += 1

    return folder_name


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
           serial-model holds references to the weights in the multi-gpu model.
           '''
        if 'load' in attrname or 'save' in attrname:
           return getattr(self._smodel, attrname)
        else:
           #return Model.__getattribute__(self, attrname)
           return super(ModelMGPU, self).__getattribute__(attrname)


class BaseModel(object):
    """
    Base functionality that all models share.

    # Arguments
        targetname: list, names of targets to predict (should match h5py file keys).
        input_shape: input feature vector shape.
    """

    def __init__(self, targetname=['teff', 'logg', 'M_H'], input_shape=None):
        self.folder_name = None
        self.model_parameter_filename = 'model_parameter.json'
        self.name = ''
        self.keras_model = None
        self.mgpu_keras_model = None
        self._model_type = ''
        self._python_info = sys.version
        self._keras_ver = keras.__version__
        self._model_identifier = self.__class__.__name__
        self._tf_ver = tf.VERSION
        self.currentdir = os.getcwd()
        self.fullfilepath = None
        self.autosave = True
        self._implementation_version = '1.0'

        ### Model hyperparameters
        self.lr = 0.0007
        self.max_epochs = 60
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [8, 8]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [1024, 512]
        self.l2 = 0
        self.batch_size = 32
        self.dropout_rate = 0
        self.last_layer_activation = 'linear'

        ### Optimizer parameters
        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = 1e-08  # a small constant for numerical stability for optimization algorithm
        self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                              decay=0.0)

        ### Default parameters for callbacks
        self.early_stopping_min_delta = 0.0001
        self.early_stopping_patience = 12
        self.reduce_lr_factor = 0.5
        self.reduce_lr_epsilon = 0.0009
        self.reduce_lr_patience = 3
        self.reduce_lr_min = 0.00002
        self.callbacks = None

        ### For data augmentation
        # The following are recommended to apply beforehand (can slow down training significantly)
        self.line_regions = None
        self.segments_step = None
        self.added_noise = None
        # These can be applied on the fly, during training
        self.max_frac_zeros = None  # Maximum fraction of spectrum to randomly assign zeros to
        self.telluric_mask_file = None  # File containing tulluric line information

        # Data parameters
        self.targetname = targetname
        self.wav = None
        self.wave_grid_key = 'wave_grid'
        self.spec_name = ''
        self.input_shape = input_shape
        self.labels_shape = len(self.targetname)
        self.data_filename = ''
        self.data_folder = ''
        self.spec_name = ''
        self.mu = None
        self.sigma = None
        self.num_train = None
        self.train_idx = None
        self.val_idx = None
        self.val_size = 0.1
        self.shuffle_indices = False
        self.use_val_generator = 'auto'  # ['false', 'true', or 'auto']

        ### Others
        self.training_generator = None
        self.validation_data = None
        self.metrics = ['mean_absolute_error']
        self.loss_func = 'mean_squared_error'
        self.history = None
        self.virtual_cvslogger = None
        self.hyper_txt = None
        self.verbose = 2
        self.premade_batches = False
        self.use_multiprocessing = True
        self.num_gpu = 1

    def get_wave_grid(self):

        try:
            with h5py.File(self.data_filename, 'r') as f:
                wav = f[self.wave_grid_key][:]
        except IOError as error:
            print(error)
            print('Data file either not declared or does not exist!')
        except KeyError as error:
            print(error)
            print('h5 data file wavelength array not stored in declared key: {}'.format(self.wave_grid_key))
        except ValueError as error:
            print(error)
            print('Invalid data file name: {}'.format(self.data_filename))

        return wav

    def get_input_shape(self):

        if self.input_shape is not None:
            return self.input_shape
        else:
            if self.wav is None:
                # Load data file, pull out wave grid
                if self.data_filename is not None:
                    print('Input shape not given, attempting to retrieve from wavelength grid in h5 file: '
                          '{}...'.format(self.data_filename))
                    wave_grid = self.get_wave_grid()
                    self.wav = wave_grid
            else:
                pass

            return (len(self.wav), 1)

    def get_model_memory_usage(self):
        from keras import backend as K

        shapes_mem_count = 0
        for l in self.keras_model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(self.keras_model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(self.keras_model.non_trainable_weights)])

        number_size = 4.0
        if K.floatx() == 'float16':
            number_size = 2.0
        if K.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size * (self.batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes

    def get_mu_and_sigma(self):

        """
        Returns the mean and standard deviation of a dataset contained in an h5 file.
        :return: ndarray mu: mean of the dataset.
        :return: ndarray sigma: standard deviation of the dataset
        """

        def stats(x):

            n = 0
            S = 0.0
            m = 0.0
            total = len(self.train_idx)
            max_chunk_size = 1000
            point = total / 100
            increment = total / 20
            train_idx = sorted(self.train_idx)  # sorting is needed for indexing h5py file

            for i in range(0, total, max_chunk_size):
                # if(i % (5 * point) == 0):
                sys.stdout.write("\r[" + "=" * int(i / increment) + " " * int((total - i) / increment) + "]" + str(
                    int(i / point)) + "%")
                sys.stdout.flush()
                indices = train_idx[i:i + max_chunk_size]
                x_subset = x[indices]

                if len(x_subset) > 0:
                    for x_i in x_subset:
                        n = n + 1
                        m_prev = m
                        m = m + (x_i - m) / n
                        S = S + (x_i - m) * (x_i - m_prev)
            return m, np.sqrt(S / n)

        model_parameter_filepath = os.path.join(self.fullfilepath, self.model_parameter_filename)

        if os.path.exists(model_parameter_filepath):
            sys.stdout.write('[INFO] Acquiring mu and sigma from model parameter file...\n')
            with open(model_parameter_filepath, 'r') as f:
                datastore = json.load(f)
                mu = np.asarray(datastore['mu'])
                sigma = np.asarray(datastore['sigma'])
                print('mu: {}'.format(mu))
                print('sigma: {}'.format(sigma))
        else:
            sys.stdout.write('[INFO] Calculating mu and sigma to normalize each label...\n')
            with h5py.File(self.data_filename, 'r') as f:
                mu = []
                sigma = []
                num_targets = len(self.targetname)
                for j, target in enumerate(self.targetname):
                    sys.stdout.write('%s/%s %s\n' % (j + 1, num_targets, target))
                    mu_, sigma_ = stats(f[target])
                    sys.stdout.write(' mu: %.1f, sigma: %.1f' % (mu_, sigma_))
                    sys.stdout.write('\n')
                    mu.append(mu_)
                    sigma.append(sigma_)
            mu = np.array(mu)
            sigma = np.array(sigma)

        return mu, sigma

    def load_validation_dataset(self, use_val_generator):
        """
        Keras has bugs with using a data generator with a validation set, often times resulting in the training
        process freezing at the end of an epoch. The validation set can be loaded directly into memory to avoid
        this issue, but of course you are free to choose if you want this.
        :param use_val_generator: ['true', 'false', 'auto'] Choose whether to use the generator, with 'auto'
                                  considering memory requirements in the decision.
        :return:
        """
        if use_val_generator.lower() == 'auto':
            print('[VALIDATION_DATA] [use_val_generator = "auto"] Validation data will either be loaded into '
                  'memory or a data generator will be used, depending on memory requirements.')

            threshold_percentage_memory = 50

            # If storing validation set in memory would occupy less than a threshold amount of free memory (set by
            # the parameter threshold_percentage_memory) then we can store in memory, else create a generator.
            num_val = len(self.val_idx)  # number of validation samples
            bytes_per_element = 8  # each element in a float array needs 8 bytes of memory
            array_len = len(self.wav)  # length of one validation sample
            target_len = len(self.targetname)
            total_bytes_val = num_val * bytes_per_element * (
                        array_len + target_len)  # total bytes required for storing validation data
            total_gb_val = total_bytes_val / 1E9  # total GB required for storing validation data

            #total_gb_model = self.get_model_memory_usage()

            mem_free_kb = int(os.popen("free | awk 'FNR == 2 {print $4}'").read())
            mem_free_gb = mem_free_kb / 1E6

            print('[VALIDATION_DATA] Validation data needs {:.1f} GB to be stored in memory.'.format(total_gb_val))
           # print('[VALIDATION_DATA] Model needs {:.1f} GB to be stored in memory.'.format(total_gb_model))
            print('[VALIDATION_DATA] Free memory available (GB): {:.1f}'.format(mem_free_gb))

            if ((total_gb_val / mem_free_gb) * 100) < threshold_percentage_memory:
                print('[VALIDATION_DATA] Memory requirements less than {}% of free memory: '
                      'loading validation data into memory'.format(threshold_percentage_memory))

                validation_data = self.load_validation_dataset(use_val_generator='false')

            else:
                print('[VALIDATION_DATA] Memory requirements greater than {}% of free memory: '
                      'using a validation data generator'.format(threshold_percentage_memory))
                validation_data = DataGenerator(indices=self.val_idx,
                                                data_filename=self.data_filename,
                                                wav=self.wav,
                                                targetname=self.targetname,
                                                mu=self.mu,
                                                sigma=self.sigma,
                                                spec_name=self.spec_name,
                                                max_added_noise=self.added_noise,
                                                line_regions=self.line_regions,
                                                segments_step=self.segments_step,
                                                max_fraction_zeros=self.max_frac_zeros,
                                                batch_size=self.batch_size,
                                                telluric_mask_file=self.telluric_mask_file)
        elif use_val_generator.lower() == 'true':
            validation_data = DataGenerator(indices=self.val_idx,
                                            data_filename=self.data_filename,
                                            wav=self.wav,
                                            targetname=self.targetname,
                                            mu=self.mu,
                                            sigma=self.sigma,
                                            spec_name=self.spec_name,
                                            max_added_noise=self.added_noise,
                                            line_regions=self.line_regions,
                                            segments_step=self.segments_step,
                                            max_fraction_zeros=self.max_frac_zeros,
                                            batch_size=self.batch_size,
                                            telluric_mask_file=self.telluric_mask_file)

        elif use_val_generator.lower() == 'false':

            print('[VALIDATION_DATA] Loading batch...')
            if self.shuffle_indices:
                print('[WARNING] Indices were shuffled, therefore indexing the h5 file might take a long time'
                      '... recommend setting shuffle_indices=False')
                X_val, y_val = load_batch_from_h5(self.data_filename,
                                                  indices=self.val_idx,
                                                  spec_name=self.spec_name,
                                                  targetname=self.targetname,
                                                  mu=self.mu, sigma=self.sigma)
            else:
                X_val, y_val = load_contiguous_slice_from_h5(self.data_filename,
                                                             start_indx=self.val_idx[0],
                                                             end_indx=self.val_idx[-1],
                                                             spec_name=self.spec_name,
                                                             targetname=self.targetname,
                                                             mu=self.mu, sigma=self.sigma)

            # Inject zeros into the spectrum
            if self.max_frac_zeros is not None:
                num_zeros = int(len(self.wav) * self.max_frac_zeros)
                print('[VALIDATION_DATA] Injecting a maximum of {} zeros into each spectrum...'.format(num_zeros))
                X_val = add_zeros(X_val, num_zeros)

            print('[VALIDATION_DATA] Applying telluric mask...')
            if self.telluric_mask_file is not None:
                if self.wav is not None:
                    telluric_mask_ = telluric_mask(self.telluric_mask_file, self.wav)
                    X_val *= telluric_mask_
                else:
                    raise ValueError('Must supply wavelength array if masking tellurics!')

            print('[VALIDATION_DATA] Reshaping...')
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

            validation_data = (X_val, y_val)

        return validation_data


    def load_pretrained_model(self, model_path):

        # Load model weights
        model = self.model()
        model.load_weights(model_path)
        self.keras_model = model

        # If there is already a training log, collect latest learning rate
        self.fullfilepath = os.path.join(self.currentdir, self.folder_name)
        training_log_path = os.path.join(self.fullfilepath, 'training.log')
        if os.path.exists(training_log_path):
            with open(training_log_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                learning_rates = []
                for row in csv_reader:
                    learning_rates.append(float(row['lr']))
                if not np.isnan(learning_rates).all():  # Don't try to find min if all NaNs
                    self.lr = np.nanmin(learning_rates)
                print('[LOAD MODEL] Last learning rate of lr={} collected from: {}'.format(self.lr,
                                                                                           training_log_path))

        # Define optimizer
        self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2,
                              epsilon=self.optimizer_epsilon)

        # Create multi-GPU version of model if GPUs > 1
        if self.num_gpu > 1:
            self.mgpu_keras_model = ModelMGPU(self.keras_model, self.num_gpu)
            self.mgpu_keras_model.compile(loss=self.loss_func, optimizer=self.optimizer, metrics=self.metrics)
        else:
            self.keras_model.compile(loss=self.loss_func, optimizer=self.optimizer, metrics=self.metrics)

        return None

    def model(self):
        input_tensor = Input(shape=self.get_input_shape(), name='input')
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0], kernel_size=self.filter_len,
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(input_tensor)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1], kernel_size=self.filter_len,
                             kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(units=self.num_hidden[0], kernel_initializer=self.initializer, activation=self.activation,
                        kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(flattener)
        layer_4 = Dense(units=self.num_hidden[1], kernel_initializer=self.initializer, activation=self.activation,
                        kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2))(layer_3)
        layer_out = Dense(units=len(self.targetname), kernel_initializer=self.initializer,
                          activation=self.last_layer_activation, name='output')(
            layer_4)
        model = Model(inputs=input_tensor, outputs=layer_out)

        return model

    def compile_model(self):

        if self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        if self.keras_model is None:

            self.keras_model = self.model()
            if self.num_gpu > 1:
                self.mgpu_keras_model = ModelMGPU(self.keras_model, self.num_gpu)
                self.mgpu_keras_model.compile(loss=self.loss_func, optimizer=self.optimizer, metrics=self.metrics,
                                              loss_weights=None)
            else:
                self.keras_model.compile(loss=self.loss_func, optimizer=self.optimizer, metrics=self.metrics,
                                         loss_weights=None)

        return None

    def save_model_parameters(self, path):

        if os.path.exists(path):
            print('[INFO] Model parameter file already exists. Skipping...')
        else:
            data = {'id': self.__class__.__name__ if self._model_identifier is None else self._model_identifier,
                    'pool_length': self.pool_length,
                    'filterlen': self.filter_len,
                    'filternum': self.num_filters,
                    'hidden': self.num_hidden,
                    'input_data_file': self.data_filename,
                    'input': self.get_input_shape(),
                    'labels': len(self.targetname),
                    'trainsize': self.num_train,
                    'valsize': self.val_size,
                    'targetname': self.targetname,
                    'spec_key': self.spec_name,
                    'dropout_rate': self.dropout_rate,
                    'l2': self.l2,
                    'batch_size': self.batch_size,
                    'mu': self.mu.tolist(),
                    'sigma': self.sigma.tolist(),
                    'max_frac_zeros': self.max_frac_zeros,
                    'max_zeros': int(self.max_frac_zeros * len(self.wav))}

            with open(path, 'w') as f:
                json.dump(data, f, indent=4, sort_keys=True)

    def save_hyperparameters(self, path):

        txt_file_path = os.path.join(self.fullfilepath, 'hyperparameter.txt')
        self.hyper_txt = open(txt_file_path, 'w')
        self.hyper_txt.write("Model: {:s} \n".format(self.name))
        self.hyper_txt.write("Model Type: {:s} \n".format(self._model_type))
        self.hyper_txt.write("Python Version: {:s} \n".format(self._python_info))
        self.hyper_txt.write("Keras Version: {:s} \n".format(self._keras_ver))
        self.hyper_txt.write("Tensorflow Version: {:s} \n".format(self._tf_ver))
        self.hyper_txt.write("Folder Name: {:s} \n".format(self.folder_name))
        self.hyper_txt.write("Batch size: {:d} \n".format(self.batch_size))
        self.hyper_txt.write("Optimizer: {:s} \n".format(self.optimizer.__class__.__name__))
        self.hyper_txt.write("Maximum Epochs: {:d} \n".format(self.max_epochs))
        self.hyper_txt.write("Learning Rate: {:f} \n".format(self.lr))
        self.hyper_txt.write("Validation Size: {:f} \n".format(self.val_size))
        self.hyper_txt.write("Input Shape: {} \n".format(self.get_input_shape()))
        self.hyper_txt.write("Label Shape: {} \n".format(np.shape(self.targetname)))
        self.hyper_txt.write("Number of Training Data: {:d} \n".format(self.num_train))
        self.hyper_txt.write("Number of Validation Data: {:d} \n".format(int(self.val_size * self.num_train)))

    def pre_training_checklist(self):

        print('[INFO] # CPUs: {}'.format(multiprocessing.cpu_count()))
        print("[INFO] training with {} GPU...".format(self.num_gpu))

        # Ensure batch size is increased if using multiple GPUs (each GPU will get self.batch_size number of spectra)
        self.batch_size = self.batch_size * self.num_gpu if self.num_gpu > 1 else self.batch_size

        # Compile the model
        self.compile_model()

        # Only generate a folder automatically if no name provided
        if self.folder_name is None:
            self.folder_name = folder_runnum()

        # If foldername doesn't already exist, then create a directory
        if not os.path.exists(os.path.join(self.currentdir, self.folder_name)):
            os.makedirs(os.path.join(self.currentdir, self.folder_name))

        # Set filepath to this directory
        self.fullfilepath = os.path.join(self.currentdir, self.folder_name)

        # Split up training set
        self.train_idx, self.val_idx = train_test_split(np.arange(self.num_train), test_size=self.val_size,
                                                        shuffle=self.shuffle_indices)
        print('[DATA] Number of Training Data: {:d}, Number of Validation Data: {:d}'.format(len(self.train_idx),
                                                                                             len(self.val_idx)))

        # Acquire input shape
        if self.input_shape is None:
            self.input_shape = self.get_input_shape()

        wave_grid = self.get_wave_grid()  # get the wave grid
        self.mu, self.sigma = self.get_mu_and_sigma()  # get label normalization values

        # Create training set data generator
        self.training_generator = DataGenerator(indices=self.train_idx,
                                                data_filename=self.data_filename,
                                                wav=wave_grid,
                                                targetname=self.targetname,
                                                mu=self.mu,
                                                sigma=self.sigma,
                                                spec_name=self.spec_name,
                                                max_added_noise=self.added_noise,
                                                line_regions=self.line_regions,
                                                segments_step=self.segments_step,
                                                max_fraction_zeros=self.max_frac_zeros,
                                                batch_size=self.batch_size,
                                                telluric_mask_file=self.telluric_mask_file)

        self.validation_data = self.load_validation_dataset(self.use_val_generator)

        ### Create callbacks ###
        # A callback for saving the current best model during training
        filepath = os.path.join(self.fullfilepath, "weights.best.h5")
        print('[INFO] Best model will be saved to: %s' % filepath)
        checkpoint = CustomModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min', save_weights_only=False)

        # Another for reducing LR if validation loss hasn't decreased in a
        # specified number of epochs (set by patience parameter)
        reduce_lr = CustomReduceLROnPlateau(monitor='val_loss', factor=self.reduce_lr_factor,
                                            patience=self.reduce_lr_patience,
                                            min_lr=self.reduce_lr_min, mode='min', verbose=2,
                                            log_path=self.fullfilepath)
        # Another for early stopping if validation loss hasn't decreased in a
        # specified number of epochs (set by patience parameter)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=self.early_stopping_min_delta,
                                       patience=self.early_stopping_patience, verbose=1, mode='min')

        # Another for keeping track of training history
        logger_path = os.path.join(self.fullfilepath, 'training.log')
        csv_logger = CSVLogger(logger_path, append=True)

        # Set all the defined callbacks
        self.callbacks = [reduce_lr, checkpoint, early_stopping, csv_logger]

        # Save model parameters
        model_parameter_filepath = os.path.join(self.fullfilepath, self.model_parameter_filename)
        print('[INFO] Saving model parameters to: %s' % model_parameter_filepath)
        self.save_model_parameters(model_parameter_filepath)

        # Save hyperparameters
        hyperparameter_filepath = os.path.join(self.fullfilepath, 'hyperparameter.txt')
        print('[INFO] Saving model hyperparameters to: %s' % hyperparameter_filepath)
        self.save_hyperparameters(hyperparameter_filepath)

        return None

    def post_training_checklist(self, total_time=0):

        # Save the finished model
        model_savename = 'model_weights.h5'
        save_path = os.path.join(self.fullfilepath, model_savename)
        self.keras_model.save(save_path)
        print(model_savename + ' saved to {:s}'.format(save_path))

    def train(self):

        # Call the pre-training checklist
        self.pre_training_checklist()

        start_time = time.time()

        # Define the number of steps per training epoch
        if self.premade_batches:  # if each h5 file contains one batch of spectra
            steps_per_epoch = self.num_train
            validation_steps = len(self.val_idx)
        else:  # if the entire dataset is contained within one h5 file
            steps_per_epoch = self.num_train // self.batch_size
            validation_steps = len(self.val_idx) // self.batch_size

        cpus = multiprocessing.cpu_count()
        workers = cpus - 2 if cpus > 2 else 1
        max_q = 10 if workers < 10 else workers

        if self.num_gpu > 1:
            model_ = self.mgpu_keras_model
        else:
            model_ = self.keras_model

        self.history = model_.fit_generator(generator=self.training_generator,
                                                      steps_per_epoch=steps_per_epoch,
                                                      validation_data=self.validation_data,
                                                      validation_steps=validation_steps,
                                                      epochs=self.max_epochs,
                                                      verbose=self.verbose,
                                                      workers=workers,
                                                      callbacks=self.callbacks,
                                                      max_q_size=max_q,
                                                      use_multiprocessing=self.use_multiprocessing)
        end_time = time.time()
        total_time = end_time - start_time

        print('Completed Training, {:.2f}s in total'.format(total_time))

        if self.autosave is True:
            # Call the post training checklist to save parameters
            self.post_training_checklist(total_time=total_time)

        return None
