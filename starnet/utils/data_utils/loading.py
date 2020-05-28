import numpy as np
import h5py
import random
import os
from astropy.io import fits as pyfits


class load_data_from_h5(object):

    def __init__(self, data_file, spec_name, target_name, wave_grid_key=None, noise_key=None, start_indx=None,
                 end_indx=None, mu=None, sigma=None):
        """
        Creates a data object containing spectra, normalized and non-normalized spectra properties
        (e.g. teff, logg, noise, etc), and the wavelength grid
        :param data_file: (string) Path of the h5 file
        :param spec_name: (string) h5 key used to access the spectra
        :param target_name: (list of strings) h5 keys used to access the spectra labels
        :param wave_grid_key: (string) h5 key used to access the wavelength grid
        :param noise_key: (string) h5 key used to access the noise data
        :param start_indx: (int) the beginning index used for slicing the data
        :param end_indx: (int) the end index used for slicing the data
        :param mu: (list) mean of the targets, used for normalization
        :param sigma: (list) std. dev. of the targets, used for normalization
        """

        self.data_file = data_file
        self.mu = mu
        self.sigma = sigma
        self.wave_grid_key = wave_grid_key
        self.noise_key = noise_key
        self.spec_name = spec_name
        self.target_name = target_name
        self.start_indx = start_indx
        self.end_indx = end_indx

        self.X = None
        self.y = None
        self.normed_y = None
        self.noise = None
        self.wave_grid = None

        self.load()

    def load(self):

        with h5py.File(self.data_file, 'r') as f:
            if self.start_indx is not None and self.end_indx is not None:
                self.X = f[self.spec_name][self.start_indx:self.end_indx]
                y = [f[target][self.start_indx:self.end_indx] for target in self.target_name]
                if self.noise_key is not None:
                    self.noise = f[self.noise_key][self.start_indx:self.end_indx]
            else:  # Load entire dataset
                self.X = f[self.spec_name][:]
                y = [f[target][:] for target in self.target_name]
                if self.noise_key is not None:
                    self.noise = f[self.noise_key][:]

            self.y = np.stack(y, axis=1)

            if self.wave_grid_key is not None:
                self.wave_grid = f[self.wave_grid_key][:]

            if self.mu is not None and self.sigma is not None:
                self.normed_y = self.normalize(self.y, self.mu, self.sigma)
            else:
                print('only non-normalized labels will be returned!')

    @staticmethod
    def normalize(data, mu, sigma):
        
        return (data-mu)/sigma

    @staticmethod
    def denormalize(data, mu, sigma):
        
        return (data*sigma)+mu

    
# ['spectrum', 'teff', 'logg', 'M_H', 'a_M', 'C_M', 'N_M']
class load_data_from_h5turbospec(object):

    def __init__(self, data_file, batch_size, indx, l, calib=False, teaghan=False,
                 mu_std='/home/spiffical/data/spiffical/cnn_synth+real/mu_std_synth+real3.txt'):
        
        with open(mu_std,'r') as f1:
            self.mu = np.array(map(float, f1.readline().split()[0:l]))
            self.sigma = np.array(map(float, f1.readline().split()[0:l]))
        
        # load data
        F = h5py.File(data_file, 'r')
        self.X = F['spectra'][indx:indx+batch_size]
        self.y = F['model_params'][indx:indx+batch_size,:l]
        
        self.normed_y = self.normalize()
        
    def normalize(self):
        
        return (self.y-self.mu)/self.sigma
            
    def denormalize(self, data):
        
        return ((data*self.sigma)+self.mu)

    
# ['spectrum', 'teff', 'logg', 'M_H', 'a_M', 'C_M', 'N_M']
def load_batch_from_h5turbospec(data_file, num_objects, batch_size, indx, l=3, mu_std=''):

    with open(mu_std,'r') as f1:
        mu = np.array(map(float, f1.readline().split()[0:l]))
        sigma = np.array(map(float, f1.readline().split()[0:l]))
        
    # Generate list of random indices (within the relevant partition of the main data file, e.g. the
    # training set) to be used to index into data_file
    indices = random.sample(range(indx, indx+num_objects), batch_size)
    indices = np.sort(indices)
    
    # load data
    F = h5py.File(data_file, 'r')
    X = F['spectra']
    y = F['model_params']

    X = X[indices,:]
    y = y[indices,:l]
        
    # Normalize labels
    normed_y = (y-mu)/sigma
        
    # Reshape X data for compatibility with CNN
    X = X.reshape(len(X), 7214, 1)
        
    return X, normed_y


def load_contiguous_slice_from_h5(data_path, start_indx, end_indx, spec_name='',
                                  targetname=['teff', 'logg', 'M_H'], mu=None, sigma=None):
    """
    To take advantage of faster loading times, h5 files can be indexed in the following way:
    --> file['test'][0:300000]
    This simple slicing is far more efficient than, for example:
    --> file['test'][range(300000] or file['test'][0, 1, 2, 3, 4, 5, ...., 300000]
    which uses h5's "fancy indexing".

    Use this function for loading in a contiguous slice from an h5 file
    :param data_path: Path of h5 file
    :param start_indx: Beginning index of h5 file
    :param end_indx: Ending index of h5 file
    :param spec_name: Key used for storing spectra in h5 file
    :param targetname: Key used for storing labels in h5 file
    :param mu: a list containing the means of each target
    :param sigma: a list containing the sigmas of each target
    :return:
    """

    with h5py.File(data_path, 'r') as data_file:  # load data file
        X = data_file[spec_name][start_indx:end_indx]
        y = [data_file[target][start_indx:end_indx] for target in targetname]
        y = np.stack(y, axis=1)

        if (mu is not None) and (sigma is not None):
            # Normalize labels
            print('[INFO] Normalizing labels...')
            y = (y - mu) / sigma

    return X, y

    
def load_batch_from_h5(data_filename, indices, spec_name='', targetname=['teff', 'logg', 'M_H'], mu=None, sigma=None):
            
        with h5py.File(data_filename, 'r') as data_file:  # load data file
            indices = indices.tolist()
            indices.sort()
            X = data_file[spec_name][indices]
            y = [data_file[target][indices] for target in targetname]
            y = np.stack(y, axis=1)

            if (mu is not None) and (sigma is not None):
                # Normalize labels
                y = (y-mu)/sigma
        
        return X, y
    
    
def load_batch_from_vmh5batch(data_file, targetname=['teff', 'logg', 'M_H'], mu_std=''):
        
        with open(mu_std,'r') as f1:
            mu = np.array(map(float, f1.readline().split()[0:len(targetname)]))
            sigma = np.array(map(float, f1.readline().split()[0:len(targetname)]))
        
        # load data
        F = h5py.File(data_file, 'r')
        X = F['spectra_AMBRE_starnetnorm'][:]
            
        targets = []
        for target in targetname:
            targets.append(F[target])
        
        y = np.column_stack([t[:] for t in targets])

        # Normalize labels
        normed_y = (y-mu)/sigma
        
        return X, normed_y


def get_synth_wavegrid(file_path, grid_name='intrigoss'):
    """
    This function will grab the wavelength grid of the synthetic spectral grid you're working with.

    :param file_path: the path of the file which contains the wavelength grid
    :param grid_name: either phoenix, intrigoss, or ambre

    :return: Wavelength grid
    """


    if grid_name.lower() == 'intrigoss':
        # For INTRIGOSS spectra, the wavelength array is stored in the same file as the spectra
        hdulist = pyfits.open(file_path)
        wave_grid_synth = hdulist[1].data['wavelength']
    elif grid_name.lower() == 'phoenix':
        # For Phoenix spectra, the wavelength array is stored in a separate file
        hdulist = pyfits.open(file_path)
        wave_grid_synth = hdulist[0].data

        # For Phoenix, need to convert from vacuum to air wavelengths.
        # The IAU standard for conversion from air to vacuum wavelengths is given
        # in Morton (1991, ApJS, 77, 119). For vacuum wavelengths (VAC) in
        # Angstroms, convert to air wavelength (AIR) via:
        #  AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)
        vac = wave_grid_synth[0]
        wave_grid_synth = wave_grid_synth / (
                1.0 + 2.735182E-4 + 131.4182 / wave_grid_synth ** 2 + 2.76249E8 / wave_grid_synth ** 4)
        air = wave_grid_synth[0]
        print('vac: {}, air: {}'.format(vac, air))
    elif grid_name.lower() == 'ambre':
        # TODO: finish this section
        wave_grid_synth = np.genfromtxt(file_path, usecols=0)
    elif grid_name.lower() == 'ferre':
        with h5py.File(file_path, 'r') as f:
            wave_grid_synth = f['wave_grid'][:]
    elif grid_name.lower() == 'nlte':
        # Get wavelength
        spec_data = np.genfromtxt(file_path)
        wave_grid_synth = spec_data[:, 0]
    else:
        raise ValueError('{} not a valid grid name. Need to supply an appropriate spectral grid name '
                         '(phoenix, intrigoss, or ambre)'.format(grid_name))

    return wave_grid_synth


def get_synth_spec_data(file_path, grid_name='phoenix', ferre_indx=np.nan, get_flux=True):
    """
    Given the path of a spectrum file and the grid of synthetic spectra you're working with (phoenix, intrigoss, ambre),
    this function will grab the flux and stellar parameters

    :param file_path: Path of the spectrum file
    :param grid_name: Name of spectral grid (phoenix, intrigoss, ambre)

    :return: flux and stellar parameters of spectrum file
    """

    # Initialize values
    teff, logg, m_h, a_m, vt, vsini = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    flux = None

    if grid_name.lower() == 'phoenix':

        with pyfits.open(file_path) as hdulist:
            if get_flux:
                flux = hdulist[0].data
            param_data = hdulist[0].header
            teff = param_data['PHXTEFF']
            logg = param_data['PHXLOGG']
            m_h = param_data['PHXM_H']
            a_m = param_data['PHXALPHA']
            vt = param_data['PHXXI_L']
    elif grid_name.lower() == 'intrigoss':
        with pyfits.open(file_path) as hdulist:
            if get_flux:
                flux = hdulist[1].data['surface_flux']
            param_data = hdulist[0].header
            teff = param_data['TEFF']
            logg = param_data['LOG_G']
            m_h = param_data['FEH']
            a_m = param_data['ALPHA']
            vt = param_data['VT']
    elif grid_name.lower() == 'ferre':
        with h5py.File(file_path, 'r') as f:
            if ferre_indx is np.nan:
                ferre_indx = random.choice(range(0, len(f['teff'][:])))
            if get_flux:
                flux = f['spectra'][ferre_indx]
            teff = f['teff'][ferre_indx]
            logg = f['logg'][ferre_indx]
            m_h = f['fe_h'][ferre_indx]
            a_m = np.nan
            vt = np.nan
    elif grid_name.lower() == 'ambre':
        filename = os.path.basename(file_path)
        teff = float(filename[1:5])
        if filename[7] == '-':
            logg = -1 * float(filename[8:11])
        else:
            logg = float(filename[8:11])
        vt = float(filename[18:20])
        if filename[22] == '-':
            m_h = -1 * float(filename[23:27])
        else:
            m_h = float(filename[23:27])
        if filename[29] == '-':
            a_m = -1 * float(filename[30:34])
        else:
            a_m = float(filename[30:34])
        if get_flux:
            flux = np.genfromtxt(file_path, usecols=-1)
    elif grid_name.lower() == 'nlte':
        # Get header
        with open(file_path) as file:
            header_data = [next(file) for x in range(10)]

        # Get spectrum and wavelength
        if get_flux:
            spec_data = np.genfromtxt(file_path)
            flux = spec_data[:, 1]

        # Collect parameters
        param_string = header_data[1].strip('#\n')
        params = [float(s) for s in param_string.split()]
        teff, logg, m_h, vt, macro, vsini = params[0], params[1], params[2], params[3], params[4], params[5]

        # Parse out abundances
        atomic_number_string, abundance_string = header_data[8].strip('#\n').split(':')
        #atomic_numbers = [int(s) for s in atomic_number_string[1:].split()]
        abundances = [float(s) for s in abundance_string[1:].split()]

        # For this particular case, all abundances varied were alpha. Just grab first abundance
        a_m = abundances[0]

    else:
        raise ValueError('{} not a valid grid name. Need to supply an appropriate spectral grid name '
                         '(phoenix, intrigoss, ambre, ferre, or nlte)'.format(grid_name))

    # Create dictionary of values
    params = {'teff': teff,
              'logg': logg,
              'm_h': m_h,
              'a_m': a_m,
              'vt': vt,
              'vrot': vsini}

    return flux, params


