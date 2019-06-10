import numpy as np
import h5py
import random

# Define edges of APOGEE detectors
beginSpecBlue = 322
endSpecBlue = 3242
beginSpecGreen = 3648
endSpecGreen = 6048   
beginSpecRed = 6412
endSpecRed = 8306

    
class load_data_from_h5(object):

    def __init__(self, data_file, specname, targetname, indices=None, mu_std=None):

        F = h5py.File(data_file, 'r')


        if indices is not None:
            indices = indices.tolist()
            X = F[specname][indices]
            y = [F[target][indices] for target in targetname]
            try:
                noise = F['noise'][indices]
            except:
                noise = None
        else:
            X = F[specname][:]
            y = [F[target][:] for target in targetname]
               
        self.y = np.stack(y, axis=1)
        self.X = X
        self.noise = noise
        try:
            self.wave_grid = F['wave_grid'][:]
        except:
            self.wave_grid = None

            self.normed_y = self.normalize(self.y)
        else:
            self.normed_y = None
        
    def normalize(self, data):
        
        return (data-self.mu)/self.sigma
            
    def denormalize(self, data):
        
        return ((data*self.sigma)+self.mu)
    
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


def get_synth_wavegrid(grid_name, synth_wave_filepath):
    """
    This function will grab the wavelength grid of the synthetic spectral grid you're working with.

    :param grid_name: either phoenix, intrigoss, or ambre
    :param synth_wave_filepath: the path of the file which contains the wavelength grid

    :return: Wavelength grid
    """

    if grid_name == 'intrigoss':
        # For INTRIGOSS spectra, the wavelength array is stored in the same file as the spectra
        hdulist = pyfits.open(synth_wave_filepath)
        wave_grid_synth = hdulist[1].data['wavelength']
    elif grid_name == 'phoenix':
        # For Phoenix spectra, the wavelength array is stored in a separate file
        hdulist = pyfits.open(synth_wave_filepath)
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
    elif grid_name == 'ambre':
        # TODO: finish this section
        wave_grid_synth = np.genfromtxt(synth_wave_filepath, usecols=0)
    else:
        raise ValueError('{} not a valid grid name. Need to supply an appropriate spectral grid name '
                         '(phoenix, intrigoss, or ambre)'.format(grid_name))

    return wave_grid_synth


def get_synth_spec_data(file_path, grid_name='phoenix'):
    """
    Given the path of a spectrum file and the grid of synthetic spectra you're working with (phoenix, intrigoss, ambre),
    this function will grab the flux and stellar parameters

    :param file_path: Path of the spectrum file
    :param grid_name: Name of spectral grid (phoenix, intrigoss, ambre)

    :return: flux and stellar parameters of spectrum file
    """

    if grid_name == 'phoenix':
        with pyfits.open(file_path) as hdulist:
            flux = hdulist[0].data
            param_data = hdulist[0].header
            teff = param_data['PHXTEFF']
            logg = param_data['PHXLOGG']
            m_h = param_data['PHXM_H']
            a_m = param_data['PHXALPHA']
            vt = param_data['PHXXI_L']
            params = [teff, logg, m_h, a_m, vt]
    elif grid_name == 'intrigoss':
        with pyfits.open(file_path) as hdulist:
            flux = hdulist[1].data['surface_flux']
            param_data = hdulist[0].header
            teff = param_data['TEFF']
            logg = param_data['LOG_G']
            m_h = param_data['FEH']
            a_m = param_data['ALPHA']
            vt = param_data['VT']
            params = [teff, logg, m_h, a_m, vt]
    elif grid_name == 'ambre':
        # TODO: finish this
        flux = np.genfromtxt(file_path, usecols=2)

        # The filename can be used to acquire the stellar parameters
        filename = os.path.basename(file_path)


    return flux, params


