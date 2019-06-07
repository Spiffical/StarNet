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
        