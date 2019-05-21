import os, sys
sys.path.insert(0, os.getenv('HOME'))
import numpy as np
import time
import warnings
import random
import glob
import h5py
import argparse
import multiprocessing
import uuid

from starnet.data.utilities.data_augmentation import add_noise, continuum_normalize, continuum_normalize_parallel, \
    add_zeros, convolve_spectrum, fastRotBroad, rotBroad, add_radial_velocity, rebin
from operator import itemgetter
from multiprocessing import Pool as ThreadPool
from astropy.io import fits as pyfits

warnings.filterwarnings('ignore')

# Define parameters needed for continuum fitting
LINE_REGIONS = [[4210, 4240], [4250, 4410], [4333, 4388], [4845, 4886], [5160, 5200], [5874, 5916], [6530, 6590]]
SEGMENTS_STEP = 10. # divide the spectrum into segments of 10 Angstroms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_dir', type=str, required=True,
                        help='location where raw spectra are stored')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='temporary location where processed spectra will be saved')
    parser.add_argument('--obs_wave_file', type=str, required=True,
                        help='path of observational wavelength grid')
    parser.add_argument('--max_num_spec', type=int, required=True,
                        help='max number of spectra to store in save_dir')
    parser.add_argument('--synth_wave_file', type=str, default=None,
                        help='path of synthetic wavelength grid (only needed if in separate file)')
    parser.add_argument('-g', '--grid', type=str, default='phoenix',
                        help='name of spectral grid to be used')
    parser.add_argument('-ho', '--host', type=str, default=None,
                        help='if transferring to remote server (not vospace), define username')
    parser.add_argument('--ip', type=str, default=None,
                        help='if transferring to remote server (not vospace), define ip address')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='number of spectra per batch')
    parser.add_argument('-n', '--noise', type=float, default=0.07,
                        help='maximum fractional noise')
    parser.add_argument('-r', '--resolution', type=int, default=47000,
                        help='instrumental resolution to convolve to')
    parser.add_argument('-vrad', '--radial_vel', type=float, default=200.,
                        help='maximum radial velocity (km/s)')
    parser.add_argument('-vrot', '--rotational_vel', type=float, default=70.,
                        help='maximum rotational velocity (km/s)')
    parser.add_argument('--max_teff', type=float, default=np.Inf,
                        help='maximum temperature (K)')
    parser.add_argument('--min_teff', type=float, default=-np.Inf,
                        help='minimum temperature (K)')
    parser.add_argument('--max_logg', type=float, default=np.Inf,
                        help='maximum logg')
    parser.add_argument('--min_logg', type=float, default=-np.Inf,
                        help='minimum logg')
    parser.add_argument('--max_feh', type=float, default=np.Inf,
                        help='maximum [Fe/H]')
    parser.add_argument('--min_feh', type=float, default=-np.Inf,
                        help='minimum [Fe/H]')

    # Collect arguments
    return parser.parse_args()


def modify_spectrum(flux, wav, new_wav, rot=65, noise=0.02, vrad=200., from_res=300000, to_res=20000, i=0):

    # Degrade resolution
    err = np.zeros(len(flux))
    _, flux, _ = convolve_spectrum(wav, flux, err, from_resolution=from_res, to_resolution=to_res)

    # Apply rotational broadening
    if rot != 0:
        epsilon = random.uniform(0, 1.)
        flux = fastRotBroad(wav, flux, epsilon, rot)

    # Add radial velocity
    if vrad != 0:
        wav = add_radial_velocity(wav, vrad)

    # Rebin to new wave grid
    flux = rebin(wav, new_wav, flux)

    # Add noise
    flux = add_noise(flux, noise)

    return flux, i


def modify_spectra_parallel(spectra, wvl, wave_grid_obs, vrot_list, noise_list, vrad_list, instrument_res, batch_size=32):

    # Make the pool of workers for parallel processing
    if batch_size < multiprocessing.cpu_count():
        pool = ThreadPool(batch_size)
    else:
        pool = ThreadPool(multiprocessing.cpu_count())

    # Modify spectra in parallel
    results = [pool.apply_async(modify_spectrum, args=(spectra[i], wvl, wave_grid_obs,
                                                       vrot_list[i], noise_list[i], vrad_list[i],
                                                       None, instrument_res, i)) for i in range(len(spectra))]
    answers = [result.get() for result in results]

    # Ensure order is preserved (pool.apply_async doesn't guarantee this)
    spectra = sorted(answers, key=itemgetter(1))
    spectra = [t[0] for t in spectra]

    return spectra


def generate_batch(file_list, wave_grid_synth, wave_grid_obs, instrument_res, batch_size=32, max_vrot=70, max_vrad=200,
                   max_noise=0.07, spectral_grid_name='phoenix', max_teff=np.Inf, min_teff=-np.Inf,
                   max_logg=np.Inf, min_logg=-np.Inf, max_feh=np.Inf, min_feh=-np.Inf):

    # Initialize lists
    spectra = []
    teff_list = []
    logg_list = []
    vt_list = []
    m_h_list = []
    a_m_list = []
    vrot_list = []
    vrad_list = []
    noise_list = []

    # Based on the observed wavelength grid, define a wavelength range to slice the synthetic wavelength grid and spectrum
    # (for quicker processing times). Extend on both sides to accommodate radial velocity shifts on synthetic spectra,
    # and also to eliminate edge effects from rotational broadening
    extension = 10  # Angstroms
    wave_min_request = wave_grid_obs[0] - extension
    wave_max_request = wave_grid_obs[-1] + extension

    wave_indices = (wave_grid_synth > wave_min_request) & (wave_grid_synth < wave_max_request)

    # Collect data for batch_size number of raw synthetic spectra
    t1 = time.time()
    while np.shape(spectra)[0]<batch_size:
        # Randomly collect a batch from the dataset
        batch_index = random.choice(range(0, len(file_list)))
        batch_file = file_list[batch_index]

        # Collect spectrum from file
        # data = np.genfromtxt(str(filename), skip_header=indx_beg, skip_footer=indx_end, usecols=(0,2))
        hdulist = pyfits.open(batch_file)

        if spectral_grid_name == 'phoenix':
            flux_ = hdulist[0].data
            param_data = hdulist[0].header
            teff = param_data['PHXTEFF']
            logg = param_data['PHXLOGG']
            m_h = param_data['PHXM_H']
            a_m = param_data['PHXALPHA']
            vt = param_data['PHXXI_L']
        elif spectral_grid_name == 'intrigoss':
            flux_ = hdulist[1].data['surface_flux']
            param_data = hdulist[0].header
            teff = param_data['TEFF']
            logg = param_data['LOG_G']
            m_h = param_data['FEH']
            a_m = param_data['ALPHA']
            vt = param_data['VT']
        elif spectral_grid_name == 'ambre':
            # TODO: fill this out
            # flux_ = data[:,1]
            # s = os.path.basename(filename)

        # Skip this spectrum if beyond requested temperature, logg, or metallicity limits
        if (teff > max_teff) or (teff < min_teff) or (logg > max_logg) or (logg < min_logg) or \
                (m_h > max_feh) or (m_h < min_feh):
            continue
        else:
            vrad = random.uniform(-max_vrad, max_vrad)  # km/s
            vrot = random.uniform(0, max_vrot)  # km/s
            noise = np.random.rand() * max_noise

            spectra.append(flux_[wave_indices])
            teff_list.append(teff)
            logg_list.append(logg)
            m_h_list.append(m_h)
            a_m_list.append(a_m)
            vt_list.append(vt)
            vrot_list.append(vrot)
            vrad_list.append(vrad)
            noise_list.append(noise)
        hdulist.close()
    t2 = time.time()
    print('Time taken to collect spectra: %.1f s' % (t2 - t1))

    # First check if wavelength array is evenly spaced. If not, modify it to be.
    wvl = wave_grid_synth[wave_indices]
    sp = wvl[1::] - wvl[0:-1]
    sp = np.append(abs(wvl[0] - wvl[1]), sp)
    sp = sp.round(decimals=5)

    unique_vals, ind, counts = np.unique(sp, return_index=True, return_counts=True)

    if len(unique_vals) > 1:  # Wavelength array is sampled differently throughout

        # Rebin fluxes to the coarsest sampling found in the wavelength grid
        coarsest_sampling = max(unique_vals[counts > 1])
        new_wave_grid = np.arange(wvl[0], wvl[-1], coarsest_sampling)

        for i, spec in enumerate(spectra):
            mod_spec = rebin(wvl, new_wave_grid, spec)
            spectra[i] = mod_spec

        wvl = new_wave_grid

    # Modify spectra in parallel (degrade resolution, apply rotational broadening, etc.)
    t1 = time.time()
    spectra = modify_spectra_parallel(spectra, wvl, wave_grid_obs, vrot_list, noise_list, vrad_list, instrument_res,
                                      batch_size)
    print('Total modify time: %.1f s' % (time.time() - t1))

    # Continuum normalize spectra with asymmetric sigma clipping continuum fitting method
    t1 = time.time()
    spectra_asym_sigma = continuum_normalize_parallel(spectra, LINE_REGIONS, wave_grid_obs, SEGMENTS_STEP,
                                                      'asym_sigmaclip', 2.0, 0.5, batch_size)

    # Continuum normalize spectra with corrected sigma clipping continuum fitting method
    spectra_c_sigma = continuum_normalize_parallel(spectra, LINE_REGIONS, wave_grid_obs, SEGMENTS_STEP,
                                                  'c_sigmaclip', 2.0, 0.5, batch_size)
    print('Total continuum time: %.2f s' % (time.time() - t1))

    params = teff_list, logg_list, m_h_list, a_m_list, vt_list, vrot_list, vrad_list, noise_list
    return spectra_asym_sigma, spectra_c_sigma, params


def main():

    # Collect arguments
    args = parse_args()

    # Collect file list
    file_list = glob.glob(os.path.join(args.spec_dir, '*.fits'))

    # Get observational and synthetic wavelength grids
    wave_grid_obs = np.load(args.wave_grid_obs_file)
    if args.grid == 'intrigoss':
        hdulist = pyfits.open(file_list[0])
        wave_grid_synth = hdulist[1].data['wavelength']
    elif args.grid == 'phoenix':
        try:
            hdulist = pyfits.open(args.wave_grid_synth_file)
            wave_grid_synth = hdulist[0].data

            # For Phoenix, need to convert from vacuum to air wavelengths.
            # The IAU standard for conversion from air to vacuum wavelengths is given
            # in Morton (1991, ApJS, 77, 119). For vacuum wavelengths (VAC) in
            # Angstroms, convert to air wavelength (AIR) via:
            #  AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)
            vac = wave_grid_synth[0]
            wave_grid_synth = wave_grid_synth / (
                    1.0 + 2.735182E-4 + 131.4182 / wave_grid_synth**2 + 2.76249E8 / wave_grid_synth**4)
            air = wave_grid_synth[0]
            print('vac: {}, air: {}'.format(vac, air))
        except:
            print('Need to supply script with a synthetic wavelength file')
    elif args.grid == 'ambre':
        # TODO: fix this section
        wave_grid_synth = np.genfromtxt(file_list[0], usecols=0)
        wave_min = wave_grid_synth[0]
        wave_max = wave_grid_synth[-1]
        dw = wave_grid_synth[1] - wave_grid_synth[0]
    else:
        raise ValueError('{} not a valid grid name. Need to supply an appropriate spectral grid name '
                         '(phoenix, intrigoss, or ambre)'.format(args.grid))

    # Determine total number of spectra already in chosen directory
    total_num_spec = 0
    existing_files = glob.glob(args.save_dir + '/*')
    if len(existing_files) > 0:
        for f in existing_files:
            f = os.path.basename(f)
            num_spec_in_file = int(f[37:])  # Number of spectra appended on to end of filename
            total_num_spec += num_spec_in_file

    print('Number of spectra already processed: {}/{}'.format(total_num_spec, args.max_num_spec))

    # Generate the batches of spectra
    spectra_created = 0
    while total_num_spec <= args.max_num_spec:

        if total_num_spec >= args.max_num_spec:
            print('Maximum number of spectra reached.')
            break

        # Process of a batch of raw synthetic spectra
        t0_batch = time.time()
        print('Creating a batch of spectra...')
        spec_asym, spec_c, params = generate_batch(file_list, wave_grid_synth, wave_grid_obs, args.resolution,
                                                   batch_size=args.batch_size,
                                                   max_vrot=args.rotational_vel,
                                                   max_vrad=args.radial_vel,
                                                   max_noise=args.noise,
                                                   spectral_grid_name=args.spectral_grid_name,
                                                   max_teff=args.max_teff,
                                                   min_teff=args.min_teff,
                                                   max_logg=args.max_logg,
                                                   min_logg=args.min_logg,
                                                   max_feh=args.max_feh,
                                                   min_feh=args.min_feh)
        teff, logg, m_h, a_m, vt, vrot, vrad, noise = params

        # Save this batch to an h5 file in chosen save directory
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        unique_filename = str(uuid.uuid4()) + '_{}'.format(args.batch_size)
        save_path = os.path.join(args.save_dir, unique_filename)
        print('Saving {} to {}'.format(unique_filename, args.save_dir))
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('spectra_starnetnorm', data=np.asarray(spec_asym))
            f.create_dataset('spectra_statcont', data=np.asarray(spec_c))
            f.create_dataset('teff', data=np.asarray(teff))
            f.create_dataset('logg', data=np.asarray(logg))
            f.create_dataset('M_H', data=np.asarray(m_h))
            f.create_dataset('a_M', data=np.asarray(a_m))
            f.create_dataset('VT', data=np.asarray(vt))
            f.create_dataset('v_rot', data=np.asarray(vrot))
            f.create_dataset('v_rad', data=np.asarray(vrad))
            f.create_dataset('noise', data=np.asarray(noise))
            f.create_dataset('wave_grid', data=np.asarray(wave_grid_obs))
        print('Total time to make this batch of {} spectra: {:.1f}'.format(args.batch_size, time.time() - t0_batch))

        spectra_created += args.batch_size
        print('Total # of spectra created so far: {}'.format(spectra_created))

        # Again, check to see how many files are in the chosen save directory (parallel jobs will be filling it up too)
        total_num_spec = 0
        existing_files = glob.glob(args.save_dir + '/*')
        for f in existing_files:
            f = os.path.basename(f)
            num_spec_in_file = int(f[37:])  # Number of spectra appended on to end of filename
            total_num_spec += num_spec_in_file
        print('Total # of spectra in directory: {}/{}'.format(total_num_spec, args.max_num_spec))


if __name__ == '__main__':
    main()
