import os, sys
sys.path.insert(0, os.path.join(os.getenv('HOME'), 'StarNet'))

import keras
import time
import numpy as np

from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer

from starnet.utils.data_utils.augment import add_noise, add_zeros, \
    add_zeros_global_error, telluric_mask
from starnet.utils.data_utils.loading import load_batch_from_h5

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, indices, data_filename, targetname, mu, sigma, spec_name, min_flux_value=0, 
                 max_flux_value=None, max_added_noise=None, max_fraction_zeros=None,
                 err_indices=None, line_regions=None, segments_step=None, wav=None, telluric_mask_file=None, 
                 batch_size=32):
        'Initialization'
        
        # Parameters for loading data
        self.data_filename = data_filename
        self.spec_name = spec_name
        self.targetname = targetname
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.indices = indices

        # Data augmentation stuff
        self.max_added_noise = max_added_noise
        self.max_fraction_zeros = max_fraction_zeros
        self.err_indices = err_indices
        self.line_regions = line_regions
        self.segments_step = segments_step
        self.min_flux_value = min_flux_value
        self.max_flux_value = max_flux_value
        self.wav = wav
        self.max_num_zeros = int(len(wav)*max_fraction_zeros)
        
        # Telluric mask
        self.telluric_mask_file = telluric_mask_file
        self.telluric_mask = None
        if self.telluric_mask_file is not None:
            if self.wav is not None:
                self.telluric_mask = telluric_mask(self.telluric_mask_file, self.wav)
            else:
                raise ValueError('Must supply wavelength array if masking tellurics!')
        
        # Generator stuff
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indices) // self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        indices_batch = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices_batch)
        
        return X, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def augment_spectra(self, X):
        
        # Add noise to spectra if desired
        if self.max_added_noise is not None:
            X = add_noise(X, self.max_added_noise)
           
        X = np.asarray(X)
        
        # Add zeroes according to a global error array
        if self.err_indices is not None:
            X = add_zeros_global_error(X, self.err_indices)
        
        # Zero out flux values below and above certain thresholds
        if self.max_flux_value is not None:
            X[X > self.max_flux_value]=0
        if self.min_flux_value is not None:
            X[X < self.min_flux_value]=0
        
        # Inject zeros into the spectrum
        if self.max_fraction_zeros is not None:
            X = add_zeros(X, self.max_num_zeros)
                
        if self.telluric_mask_file is not None and self.telluric_mask is None:
            if self.wav is not None:
                self.telluric_mask = telluric_mask(self.telluric_mask_file, self.wav)
                X *= self.telluric_mask
            else:
                raise ValueError('Must supply wavelength array if masking tellurics!')
        elif self.telluric_mask is not None:
            X *= self.telluric_mask
            
        return X

    def __data_generation(self, indices):
        'Generates data containing batch_size samples'

        x_batch, y_batch = load_batch_from_h5(data_filename=self.data_filename,
                                              indices=indices,
                                              targetname=self.targetname,
                                              mu=self.mu,
                                              sigma=self.sigma,
                                              spec_name=self.spec_name)

        x_batch = self.augment_spectra(x_batch)
        
        # Reshape data for network
        x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1], 1)

        return x_batch, y_batch


class DataGenerator_HardMining(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, model, indices, data_filename, targetname, mu, sigma, spec_name, min_flux_value=0,
                 max_flux_value=None, max_added_noise=None, max_fraction_zeros=None,
                 err_indices=None, line_regions=None, segments_step=None, wav=None, telluric_mask_file=None,
                 batch_size=32):
        'Initialization'

        # Parameters for loading data
        self.model = model
        self.data_filename = data_filename
        self.spec_name = spec_name
        self.targetname = targetname
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.indices = indices

        # Data augmentation stuff
        self.max_added_noise = max_added_noise
        self.max_fraction_zeros = max_fraction_zeros
        self.err_indices = err_indices
        self.line_regions = line_regions
        self.segments_step = segments_step
        self.min_flux_value = min_flux_value
        self.max_flux_value = max_flux_value
        self.wav = wav
        self.max_num_zeros = int(len(wav) * max_fraction_zeros)

        # Telluric mask
        self.telluric_mask_file = telluric_mask_file
        self.telluric_mask = None
        if self.telluric_mask_file is not None:
            if self.wav is not None:
                self.telluric_mask = telluric_mask(self.telluric_mask_file, self.wav)
            else:
                raise ValueError('Must supply wavelength array if masking tellurics!')

        # Generator stuff
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indices) // self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indices of the batch
        indices_temp = np.copy(self.indices)
        np.random.shuffle(indices_temp)
        hardmine_index = 0
        length_indices = len(self.indices)

        t0 = time.time()
        samples, targets = [], []
        while len(samples) < self.batch_size:
            if hardmine_index > length_indices:
                hardmine_index = 0

            indices_batch = indices_temp[hardmine_index * self.batch_size:(hardmine_index+1) * self.batch_size]

            # Generate data
            x_data, y_data = self.__data_generation(indices_batch)

            preds = self.model.predict(x_data)

            errors = np.abs(preds - y_data).max(axis=-1) > .65
            samples += x_data[errors].tolist()
            targets += y_data[errors].tolist()
        t_final = time.time() - t0
        print('Mining hard samples took {:.2f} seconds'.format(t_final))

        regular_samples = self.batch_size * 2 - len(samples)
        indices_batch = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        x_data, y_data = self.__data_generation(indices_batch)
        samples += x_data[:regular_samples].tolist()
        targets += y_data[:regular_samples].tolist()

        samples, targets = map(np.array, (samples, targets))

        idx = np.arange(self.batch_size * 2)
        np.random.shuffle(idx)
        batch = idx[:self.batch_size]

        batch_samples = samples[batch]
        batch_targets = targets[batch]

        return batch_samples, batch_targets

    def on_epoch_end(self):
        'Updates indices after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indices)

    def augment_spectra(self, X):

        # Add noise to spectra if desired
        if self.max_added_noise is not None:
            X = add_noise(X, self.max_added_noise)

        X = np.asarray(X)

        # Add zeroes according to a global error array
        if self.err_indices is not None:
            X = add_zeros_global_error(X, self.err_indices)

        # Zero out flux values below and above certain thresholds
        if self.max_flux_value is not None:
            X[X > self.max_flux_value] = 0
        if self.min_flux_value is not None:
            X[X < self.min_flux_value] = 0

        # Inject zeros into the spectrum
        if self.max_fraction_zeros is not None:
            X = add_zeros(X, self.max_num_zeros)

        if self.telluric_mask_file is not None and self.telluric_mask is None:
            if self.wav is not None:
                self.telluric_mask = telluric_mask(self.telluric_mask_file, self.wav)
                X *= self.telluric_mask
            else:
                raise ValueError('Must supply wavelength array if masking tellurics!')
        elif self.telluric_mask is not None:
            X *= self.telluric_mask

        return X

    def __data_generation(self, indices):
        'Generates data containing batch_size samples'

        x_batch, y_batch = load_batch_from_h5(data_filename=self.data_filename,
                                              indices=indices,
                                              targetname=self.targetname,
                                              mu=self.mu,
                                              sigma=self.sigma,
                                              spec_name=self.spec_name)

        x_batch = self.augment_spectra(x_batch)

        # Reshape data for network
        x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1], 1)

        return x_batch, y_batch

    def custom_generate_batch(self):

        batch_indices = self.indices[:self.batch_size]

        x_batch, y_batch = load_batch_from_h5(data_filename=self.data_filename,
                                              indices=batch_indices,
                                              targetname=self.targetname,
                                              mu=self.mu,
                                              sigma=self.sigma,
                                              spec_name=self.spec_name)

        x_batch = self.augment_spectra(x_batch)

        # Reshape data for network
        x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1], 1)

        return x_batch, y_batch


class DataGeneratorDeepEnsemble(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, indices, data_filename, targetname, mu, sigma, spec_name, min_flux_value=0,
                 max_flux_value=1.03, max_added_noise=None, max_fraction_zeros=None,
                 err_indices=None, line_regions=None, segments_step=None, wav=None, telluric_mask_file=None,
                 batch_size=32):
        'Initialization'

        # Parameters for loading data
        self.data_filename = data_filename
        self.spec_name = spec_name
        self.targetname = targetname
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.indices = indices

        # Data augmentation stuff
        self.max_added_noise = max_added_noise
        self.max_fraction_zeros = max_fraction_zeros
        self.err_indices = err_indices
        self.line_regions = line_regions
        self.segments_step = segments_step
        self.min_flux_value = min_flux_value
        self.max_flux_value = max_flux_value
        self.wav = wav
        self.max_num_zeros = int(len(wav) * max_fraction_zeros)

        # Telluric mask
        self.telluric_mask_file = telluric_mask_file
        self.telluric_mask = None
        if self.telluric_mask_file is not None:
            if self.wav is not None:
                self.telluric_mask = telluric_mask(self.telluric_mask_file, self.wav)
            else:
                raise ValueError('Must supply wavelength array if masking tellurics!')

        # Generator stuff
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indices) // self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        indices_temp = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indices)

    def augment_spectra(self, X):

        # Add noise to spectra if desired
        if self.max_added_noise is not None:
            X = add_noise(X, self.max_added_noise)

        X = np.asarray(X)

        # Add zeroes according to a global error array
        if self.err_indices is not None:
            X = add_zeros_global_error(X, self.err_indices)

        # Zero out flux values below and above certain thresholds
        if self.max_flux_value is not None:
            X[X > self.max_flux_value] = 0
        if self.min_flux_value is not None:
            X[X < self.min_flux_value] = 0

        # Inject zeros into the spectrum
        if self.max_fraction_zeros is not None:
            X = add_zeros(X, self.max_num_zeros)

        if self.telluric_mask_file is not None and self.telluric_mask is None:
            if self.wav is not None:
                self.telluric_mask = telluric_mask(self.telluric_mask_file, self.wav)
                X *= self.telluric_mask
            else:
                raise ValueError('Must supply wavelength array if masking tellurics!')
        elif self.telluric_mask is not None:
            X *= self.telluric_mask

        return X

    def __data_generation(self, indices):
        'Generates data containing batch_size samples'
        x_batch, y_batch = load_batch_from_h5(data_filename=self.data_filename,
                                              indices=indices,
                                              targetname=self.targetname,
                                              mu=self.mu,
                                              sigma=self.sigma,
                                              spec_name=self.spec_name)

        x_batch = self.augment_spectra(x_batch)

        # Reshape data for network
        x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1], 1)

        return [x_batch, y_batch], None


def mine_hard_samples(model, datagen, batch_size, use_multiprocessing, shuffle, workers, max_queue_size):

    use_sequence_api = is_sequence(datagen)
    if use_sequence_api:
        enqueuer = OrderedEnqueuer(
            datagen,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle)
    else:
        enqueuer = GeneratorEnqueuer(
            datagen,
            use_multiprocessing=use_multiprocessing)
    enqueuer.start(workers=workers, max_queue_size=max_queue_size)
    output_generator = enqueuer.get()

    while True:
        samples, targets = [], []
        while len(samples) < batch_size:
            x_data, y_data = next(output_generator)
            preds = model.predict(x_data)
            print('\nERRORS:\n{}\n'.format(np.abs(preds - y_data).max(axis=-1)))
            errors = np.abs(preds - y_data).max(axis=-1) > .99
            samples += x_data[errors].tolist()
            targets += y_data[errors].tolist()

        regular_samples = batch_size * 2 - len(samples)
        x_data, y_data = next(datagen)
        samples += x_data[:regular_samples].tolist()
        targets += y_data[:regular_samples].tolist()

        samples, targets = map(np.array, (samples, targets))

        idx = np.arange(batch_size * 2)
        np.random.shuffle(idx)
        batch1, batch2 = np.split(idx, 2)

        yield samples[batch1], targets[batch1]
        yield samples[batch2], targets[batch2]
