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

from starnet.data.utilities.generator import DataGenerator
from starnet.nn.utilities.custom_callbacks import CustomModelCheckpoint, CustomReduceLROnPlateau
from keras.callbacks import EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model, Model
from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten, Input

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


class BaseModel(object):
    """
    Base functionality that all models share.

    # Arguments
        targetname: list, names of targets to predict (should match h5py file keys).
        input_shape: input feature vector shape.
    """

    def __init__(self, targetname=['teff', 'logg', 'M_H'], input_shape=(None, 29850, 1)):
        self.folder_name = None
        self.model_parameter_filename = 'model_parameter.json'
        self.name = ''
        self.keras_model = None
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
        self.spec_norm = True
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

        ### Others
        self.training_generator = None
        self.validation_generator = None
        self.metrics = ['mean_absolute_error']
        self.loss_func = 'mean_squared_error'
        self.history = None
        self.virtual_cvslogger = None
        self.hyper_txt = None
        self.verbose = 2
        self.premade_batches = False
        self.use_multiprocessing = True

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

        if self.wav is None:
            # Load data file, pull out wave grid
            self.wav = self.get_wave_grid()

        return (len(self.wav), 1)

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
            sys.stdout.write('Acquiring mu and sigma from model parameter file...\n')
            with open(model_parameter_filepath, 'r') as f:
                datastore = json.load(f)
                mu = np.asarray(datastore['mu'])
                sigma = np.asarray(datastore['sigma'])
                print('mu: {}'.format(mu))
                print('sigma: {}'.format(sigma))
        else:
            sys.stdout.write('Calculating mu and sigma to normalize each label...\n')
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

    def load_pretrained_model(self, model):

        self.keras_model = load_model(model)
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
            self.keras_model.compile(loss=self.loss_func, optimizer=self.optimizer, metrics=self.metrics,
                                     loss_weights=None)

        return None

    def save_model_parameters(self, path):

        if os.path.exists(path):
            print('Model parameter file already exists. Skipping...')
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
        self.train_idx, self.val_idx = train_test_split(np.arange(self.num_train), test_size=self.val_size)

        # Acquire input shape
        if self.input_shape is None:
            self.input_shape = self.get_input_shape()

        wave_grid = self.get_wave_grid()  # get the wave grid
        self.mu, self.sigma = self.get_mu_and_sigma()  # get label normalization values

        # Create training and validation data generators
        self.training_generator = DataGenerator(indices=self.train_idx,
                                                data_filename=self.data_filename,
                                                wav=wave_grid,
                                                targetname=self.targetname,
                                                mu=self.mu,
                                                sigma=self.sigma,
                                                spec_name=self.spec_name,
                                                normalized_spec=self.spec_norm,
                                                max_added_noise=self.added_noise,
                                                line_regions=self.line_regions,
                                                segments_step=self.segments_step,
                                                max_fraction_zeros=self.max_frac_zeros,
                                                batch_size=self.batch_size,
                                                telluric_mask_file=self.telluric_mask_file)
        self.validation_generator = DataGenerator(indices=self.val_idx,
                                                  data_filename=self.data_filename,
                                                  wav=wave_grid,
                                                  targetname=self.targetname,
                                                  mu=self.mu,
                                                  sigma=self.sigma,
                                                  spec_name=self.spec_name,
                                                  normalized_spec=self.spec_norm,
                                                  max_added_noise=self.added_noise,
                                                  line_regions=self.line_regions,
                                                  segments_step=self.segments_step,
                                                  max_fraction_zeros=self.max_frac_zeros,
                                                  batch_size=self.batch_size,
                                                  telluric_mask_file=self.telluric_mask_file)

        ### Create callbacks ###
        # A callback for saving the current best model during training
        filepath = os.path.join(self.fullfilepath, "weights.best.h5")
        print('Best model will be saved to: %s' % filepath)
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

        print('Number of Training Data: {:d}, Number of Validation Data: {:d}'.format(len(self.train_idx),
                                                                                      len(self.val_idx)))

        # Save model parameters
        model_parameter_filepath = os.path.join(self.fullfilepath, self.model_parameter_filename)
        print('Saving model parameters to: %s' % model_parameter_filepath)
        self.save_model_parameters(model_parameter_filepath)

        # Save hyperparameters
        hyperparameter_filepath = os.path.join(self.fullfilepath, 'hyperparameter.txt')
        print('Saving model parameters to: %s' % hyperparameter_filepath)
        self.save_hyperparameters(hyperparameter_filepath)

        return None

    def post_training_checklist(self, total_time=0):

        # Save the finished model
        model_savename = 'model_weights.h5'
        save_path = os.path.join(self.fullfilepath, model_savename)
        self.keras_model.save(save_path)
        print(model_savename + ' saved to {:s}'.format(save_path))

    def train(self):
        # Call the checklist to create folder and save parameters
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
        workers = cpus - 1 if cpus > 1 else 1
        self.history = self.keras_model.fit_generator(generator=self.training_generator,
                                                      steps_per_epoch=steps_per_epoch,
                                                      validation_data=self.validation_generator,
                                                      validation_steps=validation_steps,
                                                      epochs=self.max_epochs,
                                                      verbose=self.verbose,
                                                      workers=workers,
                                                      callbacks=self.callbacks,
                                                      use_multiprocessing=self.use_multiprocessing)
        end_time = time.time()
        total_time = end_time - start_time

        print('Completed Training, {:.2f}s in total'.format(total_time))

        if self.autosave is True:
            # Call the post training checklist to save parameters
            self.post_training_checklist(total_time=total_time)

        return None
