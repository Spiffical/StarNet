# Arguments to file should be [starting index, ending index]
import sys
import os
#os.environ['TURBODATA'] = '/home/merileo/projects/rrg-kyi/merileo/turbospec'
os.environ['TURBODATA'] = '/scratch/merileo/Turbo/Turbospectrum2019/DATA'
os.environ['SDSS_LOCAL_SAS_MIRROR'] = '/home/merileo/projects/rrg-kyi/merileo/sdss'
os.environ['RESULTS_VERS'] = 'l31c.2'

import apogee.tools.path as appath
appath.change_dr(dr='12')

import turbospec
import atlas9
import numpy as np
import h5py
import random

curr_file_num_list = [0]#[int(arg) for arg in args]

BATCH_SIZE = 2

atmospheric_params = {'teff': np.zeros(BATCH_SIZE), 'logg': np.zeros(BATCH_SIZE),
                      'M_H': np.zeros(BATCH_SIZE), 'C_M': np.zeros(BATCH_SIZE),
                      'a_M': np.zeros(BATCH_SIZE), 'vmicro': np.zeros(BATCH_SIZE),
                      'vmacro': np.zeros(BATCH_SIZE), 'vrot': np.zeros(BATCH_SIZE),
                      'vrad': np.zeros(BATCH_SIZE), 'noise': np.zeros(BATCH_SIZE)}

abundances = {'Li': {'val': np.zeros(BATCH_SIZE), 'an': 3}, 'Na': {'val': np.zeros(BATCH_SIZE), 'an': 11},
              'Mg': {'val': np.zeros(BATCH_SIZE), 'an': 12}, 'Cl': {'val': np.zeros(BATCH_SIZE), 'an': 17},
              'O': {'val': np.zeros(BATCH_SIZE), 'an': 8}, 'Al': {'val': np.zeros(BATCH_SIZE), 'an': 13},
              'Si': {'val': np.zeros(BATCH_SIZE), 'an': 14}, 'P': {'val': np.zeros(BATCH_SIZE), 'an': 15},
              'S': {'val': np.zeros(BATCH_SIZE), 'an': 16}, 'Fe': {'val': np.zeros(BATCH_SIZE), 'an': 26},
              'Ca': {'val': np.zeros(BATCH_SIZE), 'an': 20}, 'Ti': {'val': np.zeros(BATCH_SIZE), 'an': 22},
              'V': {'val': np.zeros(BATCH_SIZE), 'an': 23}, 'Cr': {'val': np.zeros(BATCH_SIZE), 'an': 24},
              'Mn': {'val': np.zeros(BATCH_SIZE), 'an': 25}, 'Co': {'val': np.zeros(BATCH_SIZE), 'an': 27},
              'Ni': {'val': np.zeros(BATCH_SIZE), 'an': 28}, 'Cu': {'val': np.zeros(BATCH_SIZE), 'an': 29},
              'Ge': {'val': np.zeros(BATCH_SIZE), 'an': 32}, 'Rb': {'val': np.zeros(BATCH_SIZE), 'an': 37},
              'Nd': {'val': np.zeros(BATCH_SIZE), 'an': 60}, 'N': {'val': np.zeros(BATCH_SIZE), 'an': 7}}

synspec_array = []
synspec_nonorm_array = []

for batch in curr_file_num_list:
    for i in range(BATCH_SIZE):
        successful = False
        while not successful:
            try:
                # Randomly select parameters for the stellar model
                atmospheric_params['teff'][i] = random.uniform(3500., 7000.)
                atmospheric_params['logg'][i] = random.uniform(0., 5.)
                atmospheric_params['M_H'][i] = random.uniform(-2.5, 0.5)
                atmospheric_params['C_M'][i] = random.uniform(-1., 1.1)
                atmospheric_params['a_M'][i] = random.uniform(-1., 1.1)
                atmospheric_params['vmicro'][i] = random.uniform(0.5, 8)  # km/s
                atmospheric_params['vmacro'][i] = random.uniform(0, 12)  # km/s FWHM
                atmospheric_params['vrot'][i] = random.uniform(0, 70)  # km/s
                atmospheric_params['vrad'][i] = random.uniform(0, 200)  # km/s
                atmospheric_params['noise'][i] = random.uniform(0, 0.07)
                for el in list(abundances.keys()):
                    abundances[el]['val'][i] = random.uniform(-2., 1.1)

                # Instantiate the model atmosphere
                atm = atlas9.Atlas9Atmosphere(teff=atmospheric_params['teff'][i],
                                              logg=atmospheric_params['logg'][i],
                                              metals=atmospheric_params['M_H'][i],
                                              am=atmospheric_params['a_M'][i],
                                              cm=atmospheric_params['C_M'][i])
                successful = True
                print('Successful!')
            except:
                pass

        # Randomize isotopic ratio (0 for solar, 1 for arcturus)
        isotopes = ['solar', 'arcturus']
        iso_num = np.random.randint(2)

        # Generate a Turbospectrum spectrum, both continuum normalized and not continuum normalized
        abundances_turbospec = [[abundances[el]['an'], abundances[el]['val'][i]] for el in list(abundances.keys())]
        synspec, wav, synspec_nonorm = turbospec.synth(*abundances_turbospec,
                                                       modelatm=atm,
                                                       linelist='turbospec.20170418',
                                                       lsf='all',
                                                       #fiber=1,
                                                       #xlsf=np.linspace(-7., 7., 43),
                                                       cont='aspcap',
                                                       vmacro=atmospheric_params['vmacro'][i],
                                                       isotopes=isotopes[iso_num],
                                                       vmicro=atmospheric_params['vmicro'][i],
                                                       vrot=atmospheric_params['vrot'][i],
                                                       vrad=atmospheric_params['vrad'][i],
                                                       noise=atmospheric_params['noise'][i],
                                                       dr='12')
        print(synspec)
        synspec_array.append(synspec[0])
        synspec_nonorm_array.append(synspec_nonorm[0])
