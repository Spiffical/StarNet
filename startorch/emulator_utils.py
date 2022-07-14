import os
import sys
import h5py
import numpy as np
import torch
import random
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from .utils import get_mean_and_std, add_noise


class HDF5Dataset(data.Dataset):
    def __init__(self, in_file, labels_key, spectra_key, mean, std, n_samples, add_zeros=True,
                 remove_gaps=False, remove_arm=False, noise_addition=False, wavegrid_path='',
                 min_wvl=None, max_wvl=None):
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

        if wavegrid_path:
            self.wave_grid = np.load(wavegrid_path)
        else:
            with h5py.File(self.in_file, 'r') as f:
                self.wave_grid = f['wave_grid'][:]


    def __getitem__(self, index):

        with h5py.File(self.in_file, 'r') as f:
            labels = np.asarray([f[target][index] for target in self.labels_key])
            labels = (labels - self.mean)/self.std

            spectrum = f[self.spectra_key][index]
            spectrum[np.isnan(spectrum)] = 0
            spectrum[np.isinf(spectrum)] = 0
            #spectrum[spectrum > 2] = 0
            #spectrum[spectrum < 0] = 0

            # inject zeros
            if self.add_zeros:
                len_spec = len(spectrum)
                max_zeros = 0.15 * len_spec
                num_zeros = random.randint(0, int(max_zeros))
                indices = random.sample(range(len_spec), num_zeros)
                if len(indices) != 0:
                    spectrum[indices] = 0
            if self.noise_addition:
                spectrum = add_noise(spectrum)
            if self.min_wvl and self.max_wvl:
                wvl_indices = (self.wave_grid > self.min_wvl) & (self.wave_grid < self.max_wvl)
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
                           wavegrid_path='',
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
                                remove_arm=remove_arm, noise_addition=noise_addition, wavegrid_path=wavegrid_path,
                                min_wvl=min_wvl, max_wvl=max_wvl)
    valid_dataset = HDF5Dataset(val_data_path, labels_key=labels_key, spectra_key=spectra_key,
                                mean=mean, std=std, n_samples=len(indices_val), remove_gaps=remove_gaps,
                                remove_arm=remove_arm, noise_addition=noise_addition, wavegrid_path=wavegrid_path,
                                min_wvl=min_wvl, max_wvl=max_wvl)

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
        wave_grid = train_dataset.wave_grid
        if min_wvl and max_wvl:
            wvl_indices = (wave_grid > min_wvl) & (wave_grid < max_wvl)
            spec = spec[wvl_indices]
        #if max_wvl:
        #    wvl_indices = wave_grid < max_wvl
        #    spec = spec[wvl_indices]
        len_spec = len(spec)

    return train_loader, valid_loader, len_spec


# Function to execute a training epoch
def train_epoch_generator(NN,training_generator,optimizer,device,train_steps,loss_fn):

    NN.train()
    loss = 0
    # Passing the data through the NN
    for i, (labels, spectra) in enumerate(training_generator):
        #sys.stdout.write('{}\n'.format(i))

        x = labels
        y = spectra

        # Transfer to device
        x = x.to(device).float().view(-1, 1, np.shape(x)[1])
        y_true = y.to(device).float()

        # perform a forward pass and calculate loss
        y_pred = NN(x)
        batch_loss = loss_fn(y_pred, y_true)

        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # add batch loss to the total training loss
        loss += batch_loss

    #scheduler.step()
    #MSE = (loss / (batch_size * (n_train // batch_size))).detach().cpu().numpy()
    avgLoss = loss / train_steps
    avgLoss = avgLoss.detach().cpu().numpy()
    return avgLoss


# Function to execute a validation epoch
def val_epoch_generator(NN,valid_generator,device,val_steps,loss_fn):
    with torch.no_grad():
        NN.eval()
        loss = 0
        # Passing the data through the NN
        for labels, spectra in valid_generator:
            x = labels
            y = spectra

            # Transfer to device
            x = x.to(device).float().view(-1, 1, np.shape(x)[1])
            y_true = y.to(device).float()

            y_pred = NN(x)
            loss += loss_fn(y_pred, y_true)#*batch_size
        #MSE = (loss/(batch_size*(n_val//batch_size))).detach().cpu().numpy()
        avgLoss = loss / val_steps
        avgLoss = avgLoss.detach().cpu().numpy()

        return avgLoss
