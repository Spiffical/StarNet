import os
import sys
import h5py
import numpy as np
import torch
import random
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

def get_mean_and_std(data_file, indices, model_folder, targets):

    """
    Returns the mean and standard deviation of a dataset contained in an h5 file.
    :return: ndarray mu: mean of the dataset.
    :return: ndarray sigma: standard deviation of the dataset
    """

    def stats(x, indices):

        n = 0
        S = 0.0
        m = 0.0
        total = len(indices)
        max_chunk_size = 1000
        point = total / 100
        increment = total / 20
        indices = sorted(indices)  # sorting is needed for indexing h5py file

        for i in range(0, total, max_chunk_size):
            # if(i % (5 * point) == 0):
            sys.stdout.write("\r[" + "=" * int(i / increment) + " " * int((total - i) / increment) + "]" + str(
                int(i / point)) + "%")
            sys.stdout.flush()
            indices_chunk = indices[i:i + max_chunk_size]
            x_subset = x[indices_chunk]

            if len(x_subset) > 0:
                for x_i in x_subset:
                    n = n + 1
                    m_prev = m
                    m = m + (x_i - m) / n
                    S = S + (x_i - m) * (x_i - m_prev)
        return m, np.sqrt(S / n)

    mean_std_params_path = os.path.join(model_folder, 'mean_std_params.npy')
    if os.path.exists(mean_std_params_path):
        sys.stdout.write('[INFO] Acquiring mu and sigma from numpy file...\n')
        mean, std = np.load(mean_std_params_path)
        print('mean: {}'.format(mean))
        print('std: {}'.format(std))
    else:
        sys.stdout.write('[INFO] Calculating mean and std to normalize each label...\n')
        with h5py.File(data_file, 'r') as f:
            mean = []
            std = []
            num_targets = len(targets)
            for j, target in enumerate(targets):
                sys.stdout.write('%s/%s %s\n' % (j + 1, num_targets, target))
                mu_, sigma_ = stats(f[target], indices)
                sys.stdout.write(' mu: %.1f, sigma: %.1f' % (mu_, sigma_))
                sys.stdout.write('\n')
                mean.append(mu_)
                std.append(sigma_)
        mean = np.array(mean)
        std = np.array(std)
        sys.stdout.write('Saving mean and std file')
        np.save(mean_std_params_path, np.array([mean, std]))

    return mean, std


# Loading data
def load_data(data_file, indices, targets):
    with h5py.File(data_file, 'r') as F:
        indices = indices.tolist()
        indices.sort()
        # Loading inputs
        inputs = [F[target][indices] for target in targets]
        inputs = np.stack(inputs, axis=1)
        # Loading labels
        labels = np.array(F['spectra_asymnorm_noiseless'][indices])
    return inputs, labels


def remove_g_or_b_arm(spectrum):
    blue_spectrum = spectrum[:11880]
    green_spectrum = spectrum[11880:25880]
    red_spectrum = spectrum[25880:]
    partitioned_spectrum = [blue_spectrum, green_spectrum, red_spectrum]

    # Determine which arm to remove
    remove_arm = np.random.randint(0, 2)
    # Replace arm with noise
    partitioned_spectrum[remove_arm] = np.random.normal(1, .005,
                                                        np.shape(partitioned_spectrum[remove_arm]))

    spectrum = np.concatenate(partitioned_spectrum)

    return spectrum


def add_noise(x, max_noise=0.07):

    noise_factor = random.uniform(0, max_noise)
    x += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)

    return x


def remove_interccd_gaps(spectrum, wave_grid):
    indices_blue = (wave_grid >= 4495) & (wave_grid <= 4541)
    indices_green = (wave_grid >= 5266) & (wave_grid <= 5319)
    indices_red = (wave_grid >= 6366) & (wave_grid <= 6441)
    spectrum[indices_blue] = np.random.normal(1, .0005, np.shape(spectrum[indices_blue]))
    spectrum[indices_green] = np.random.normal(1, .0005, np.shape(spectrum[indices_green]))
    spectrum[indices_red] = np.random.normal(1, .0005, np.shape(spectrum[indices_red]))

    return spectrum


class HDF5Dataset(data.Dataset):
    def __init__(self, in_file, labels_key, spectra_key, mean, std, n_samples, add_zeros=True,
                 remove_gaps=False, remove_arm=False, noise_addition=False, min_wvl=None, max_wvl=None):
        super(HDF5Dataset, self).__init__()

        self.labels_key = labels_key
        self.spectra_key = spectra_key
        self.mean = mean
        self.std = std
        self.n_samples = n_samples
        self.in_file = in_file
        self.add_zeros = add_zeros
        self.remove_gaps = remove_gaps
        self.remove_arm = remove_arm
        self.noise_addition = noise_addition
        self.min_wvl = min_wvl
        self.max_wvl = max_wvl

    def __getitem__(self, index):

        with h5py.File(self.in_file, 'r') as f:
            labels = np.asarray([f[target][index] for target in self.labels_key])
            labels = (labels - self.mean)/self.std

            spectrum = f[self.spectra_key][index]
            spectrum[np.isnan(spectrum)] = 0
            spectrum[np.isinf(spectrum)] = 0
            spectrum[spectrum > 2] = 0
            spectrum[spectrum < 0] = 0
            wave_grid = f['wave_grid'][:]

            # inject zeros
            if self.add_zeros:
                len_spec = len(spectrum)
                max_zeros = 0.15 * len_spec
                num_zeros = random.randint(0, max_zeros)
                indices = random.sample(range(len_spec), num_zeros)
                if len(indices) != 0:
                    spectrum[indices] = 0
            if self.noise_addition:
                spectrum = add_noise(spectrum)
            if self.remove_gaps:
                spectrum = remove_interccd_gaps(spectrum, wave_grid)
            if self.remove_arm:
                spectrum = remove_g_or_b_arm(spectrum)
            if self.min_wvl and self.max_wvl:
                wvl_indices = (wave_grid > self.min_wvl) & (wave_grid < self.max_wvl)
                spectrum = spectrum[wvl_indices]
            #if self.min_wvl:
            #    wvl_indices = wave_grid > self.min_wvl
            #    spectrum = spectrum[wvl_indices]
            #if self.max_wvl:
            #    wvl_indices = wave_grid < self.max_wvl
            #    spectrum = spectrum[wvl_indices]


            #mask = np.ones_like(spectrum)
            #mask[spectrum > 3] = 0
            #mask[spectrum == 0] = 0

            #labels = torch.from_numpy(labels).to('cuda:0').float().view(-1, 1, len(self.labels_key))
            #mask = torch.from_numpy(mask).to('cuda:0').float().view(-1, 1, len(mask))
            #spectra = torch.from_numpy(spectrum).to('cuda:0').float().view(-1, 1, len(spectrum))

        return labels, spectrum

    def __len__(self):
        return self.n_samples


def get_train_valid_loader(data_path,
                           batch_size,
                           save_folder,
                           labels_key,
                           spectra_key,
                           num_train,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True,
                           remove_gaps=False,
                           remove_arm=False,
                           noise_addition=False,
                           val_data_path='',
                           min_wvl=None,
                           max_wvl=None
                           ):
    """
    Utility function for loading and returning train and valid
    multi-process iterators. A sample
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_path: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    frac_train = 1 - valid_size
    indices_reference = np.arange(num_train)
    if shuffle:
        np.random.shuffle(indices_reference)
    indices_train = indices_reference[:int(frac_train * len(indices_reference))]

    if not val_data_path:
        val_data_path = data_path
        indices_val = indices_reference[int(frac_train * len(indices_reference)):]
    else:
        indices_val = np.arange(int(frac_train * len(indices_reference)))
        np.random.shuffle(indices_val)

    # Acquire mean and std of input
    mean, std = get_mean_and_std(data_path, indices_train, save_folder, labels_key)

    # Initialize data loaders
    train_dataset = HDF5Dataset(data_path, labels_key=labels_key, spectra_key=spectra_key,
                                mean=mean, std=std, n_samples=len(indices_train), remove_gaps=remove_gaps,
                                remove_arm=remove_arm, noise_addition=noise_addition, min_wvl=min_wvl, max_wvl=max_wvl)
    valid_dataset = HDF5Dataset(val_data_path, labels_key=labels_key, spectra_key=spectra_key,
                                mean=mean, std=std, n_samples=len(indices_val), remove_gaps=remove_gaps,
                                remove_arm=remove_arm, noise_addition=noise_addition, min_wvl=min_wvl, max_wvl=max_wvl)

    train_sampler = SubsetRandomSampler(indices_train)
    valid_sampler = SubsetRandomSampler(indices_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=False,
    )

    with h5py.File(data_path, 'r') as f:
        spec = f[spectra_key][0]
        wave_grid = f['wave_grid'][:]
        if min_wvl and max_wvl:
            wvl_indices = (wave_grid > min_wvl) & (wave_grid < max_wvl)
            spec = spec[wvl_indices]
        #if max_wvl:
        #    wvl_indices = wave_grid < max_wvl
        #    spec = spec[wvl_indices]
        len_spec = len(spec)

    return train_loader, valid_loader, len_spec
