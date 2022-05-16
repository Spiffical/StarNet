import sys
import os
sys.path.insert(0, "{}/StarNet".format(os.getenv('HOME')))
import os
import numpy as np
import time
import glob
import h5py
from preprocess_spectra import continuum_normalize, continuum_normalize_parallel

from astropy.io import fits as pyfits


home = os.getenv('HOME')
uves_data_folder = os.path.join(home, 'projects/rrg-kyi/group_writable/spectra/flames-uves/')
uves_catalog_path = os.path.join(home, 'projects/rrg-kyi/group_writable/spectra/flames-uves/catalog/'
                                 'gaia-eso-catalog.fits')

# Load catalog
hdulist_catalog = pyfits.open(uves_catalog_path)

# get the data (in extension 1)
catalog_data = hdulist_catalog[1].data

# Wavelengths for masking beginning and end of spectrum
wave_min_mask = 4835
wave_max_mask = 5395

# Define parameters needed for continuum fitting
LINE_REGIONS = [[4210, 4240], [4250, 4410], [4333, 4388], [4845, 4886], [5160, 5200], [5874, 5916], [6530, 6590]]
SEGMENTS_STEP = 10.  # divide the spectrum into segments of 10 Angstroms

wave_grid_final = np.load(os.path.join(home, 'projects/rrg-kyi/group_writable/spectra/UVES_4835-5395.npy'))

# Define requested parameters
instrument_ = 'UVES'
ges_ = 'GE_MW'  # 'GE_SD_BM'
# wl_min_ = 582.2
# wl_max_ = 683.0913
wl_min_ = 476.8
wl_max_ = 580.1746

# Initiate lists
spectra = []
spectra_starnet_norm = []
spectra_starnet_norm_R10000 = []
spectra_gaussian_norm10A = []
spectra_gaussian_norm50A = []
err_spectra = []
qual_list = []
teff_list = []
teff_err_list = []
logg_list = []
logg_err_list = []
feh_list = []
feh_err_list = []
vrad_list = []
vrad_err_list = []
vmicro_list = []
vmicro_err_list = []
snr_list = []
object_list = []
ges_type_list = []
i = 0
for filename in glob.glob(uves_data_folder + '*.fits'):
    with pyfits.open(filename, memmap=False) as hdulist_spec:
        try:
            instrument = hdulist_spec[0].header['INSTRUME']
            # dispelem = hdulist_spec[0].header['DISPELEM']
            # obj = hdulist_spec[0].header['OBJECT']
            ges = hdulist_spec[0].header['GES_TYPE']
            wl_min = hdulist_spec[0].header['WAVELMIN']
            wl_max = hdulist_spec[0].header['WAVELMAX']
        except:
            continue
        if instrument == instrument_ and 'GE' in ges and wl_min == wl_min_ and wl_max == wl_max_:
            snr = hdulist_spec[0].header['SNR']
            obj = hdulist_spec[0].header['OBJECT']
            wav = hdulist_spec[1].data['WAVE'][0]
            flux = hdulist_spec[1].data['FLUX'][0]
            err_spec = hdulist_spec[1].data['ERR'][0]
            qual = hdulist_spec[1].data['QUAL'][0]
            wav = wav * 10  # Convert to Angstroms
        else:
            continue

        mask = (wav > wave_min_mask) & (wav < wave_max_mask)
        flux = flux[mask]
        err_spec = err_spec[mask]
        wav = wav[mask]
        qual = qual[mask]

        indx = np.where(hdulist_spec[0].header['OBJECT'] == catalog_data['CNAME'])[0]

        stellar_params = dict()
        stellar_params_errs = dict()
        param_labels = ['TEFF', 'LOGG', 'FEH', 'VRAD', 'XI']
        param_err = ['E_TEFF', 'E_LOGG', 'E_FEH', 'E_VRAD', 'E_XI']

        for (param, err) in zip(param_labels, param_err):

            try:
                stellar_params[param] = catalog_data[param][indx][0]
                stellar_params_errs[err] = catalog_data[err][indx][0]
            except:
                stellar_params[param] = np.nan
                stellar_params_errs[err] = np.nan
                print('no %s' % param)

        stellar_param_vals = list(stellar_params.values())
        stellar_param_err_vals = list(stellar_params_errs.values())
        teff_list.append(stellar_param_vals[0])
        teff_err_list.append(stellar_param_err_vals[0])
        logg_list.append(stellar_param_vals[1])
        logg_err_list.append(stellar_param_err_vals[1])
        feh_list.append(stellar_param_vals[2])
        feh_err_list.append(stellar_param_err_vals[2])
        vrad_list.append(stellar_param_vals[3])
        vrad_err_list.append(stellar_param_err_vals[3])
        vmicro_list.append(stellar_param_vals[4])
        vmicro_err_list.append(stellar_param_err_vals[4])
        snr_list.append(snr)
        spectra.append(flux)
        err_spectra.append(err_spec)
        qual_list.append(qual)
        object_list.append(obj)
        ges_type_list.append(ges)

        if i % 10 == 0:
            print('Collected {} spectra so far...'.format(i))
        i += 1

BATCH_SIZE = 16
# Now continuum normalize all the spectra...
t0 = time.time()
for i in range(len(spectra))[::BATCH_SIZE]:

    spectra_starnetnorm_temp = continuum_normalize_parallel(spectra[i:i+BATCH_SIZE], wav,
                                                            line_regions=LINE_REGIONS,
                                                            segments_step=SEGMENTS_STEP,
                                                            fit='asym_sigmaclip')
    spectra_gaussiannorm10A_temp = continuum_normalize(spectra[i:i + BATCH_SIZE], wav,
                                                    err=qual_list[i:i + BATCH_SIZE],
                                                    fit='gaussian_smooth', sigma_gaussian=10)
    spectra_gaussiannorm50A_temp = continuum_normalize(spectra[i:i + BATCH_SIZE], wav,
                                                       err=qual_list[i:i + BATCH_SIZE],
                                                       fit='gaussian_smooth', sigma_gaussian=50)
    print('{} spectra continuum normalized so far, taking {:.2f} seconds'.format(i+BATCH_SIZE, time.time() - t0))
    spectra_starnet_norm.extend(spectra_starnetnorm_temp)
    spectra_gaussian_norm10A.extend(spectra_gaussiannorm10A_temp)
    spectra_gaussian_norm50A.extend(spectra_gaussiannorm50A_temp)

object_list = [a.encode('utf8') for a in object_list]
ges_type_list = [a.encode('utf8') for a in ges_type_list]

save_folder = os.path.join(home, 'projects/rrg-kyi/group_writable/spectra/preprocessed/')
file_name = 'UVES_' + ges_ + '_' + str(wave_min_mask) + '-' + str(wave_max_mask) + '_final_werrors_wgaussian_wR10000.h5'
with h5py.File(save_folder + file_name, "w") as f:
    spectra_dset = f.create_dataset('spectra', data=np.asarray(spectra))
    spectra_norm_dset = f.create_dataset('spectra_asymnorm', data=np.asarray(spectra_starnet_norm))
    spectra_gaussiannorm10_dset = f.create_dataset('spectra_gaussiannorm10A', data=np.asarray(spectra_gaussian_norm10A))
    spectra_gaussiannorm50_dset = f.create_dataset('spectra_gaussiannorm50A', data=np.asarray(spectra_gaussian_norm50A))
    err_spectra_dset = f.create_dataset('error_spectra', data=np.asarray(err_spectra))
    qual_dset = f.create_dataset('qual', data=np.asarray(qual_list))
    teff_dset = f.create_dataset('teff', data=np.asarray(teff_list))
    teff_err_dset = f.create_dataset('teff_err', data=np.asarray(teff_err_list))
    logg_dset = f.create_dataset('logg', data=np.asarray(logg_list))
    logg_err_dset = f.create_dataset('logg_err', data=np.asarray(logg_err_list))
    feh_dset = f.create_dataset('fe_h', data=np.asarray(feh_list))
    feh_err_dset = f.create_dataset('fe_h_err', data=np.asarray(feh_err_list))
    vrad_dset = f.create_dataset('v_rad', data=np.asarray(vrad_list))
    vrad_err_dset = f.create_dataset('v_rad_err', data=np.asarray(vrad_err_list))
    vmicro_dset = f.create_dataset('vmicro', data=np.asarray(vmicro_list))
    vmicro_err_dset = f.create_dataset('vmicro_err', data=np.asarray(vmicro_err_list))
    snr_dset = f.create_dataset('SNR', data=np.asarray(snr_list))
    obj_dset = f.create_dataset('object', data=np.asarray(object_list))
    wave_dset = f.create_dataset('wave_grid', data=np.asarray(wav))
    ges_dset = f.create_dataset('ges_type', data=np.asarray(ges_type_list))
