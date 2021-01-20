import numpy as np
import time
import glob
import h5py
import argparse
import uuid
import os
import sys
sys.path.insert(0, os.path.join(os.getenv('HOME'), 'StarNet'))
from starnet.utils.data_utils.preprocess_spectra import preprocess_batch_of_spectra, preprocess_batch_of_aat_spectra
from starnet.utils.data_utils.loading import collect_file_list


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
    parser.add_argument('--telescope', type=str, default='',
                        help='telescope used for observations')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='number of spectra per batch')
    parser.add_argument('-n', '--noise', type=float, default=0.07,
                        help='maximum fractional noise')
    parser.add_argument('-r', '--resolution', type=int, default=47000,
                        help='instrumental resolution to convolve to')
    parser.add_argument('-vrad', '--radial_vel', type=float, default=200.,
                        help='maximum radial velocity (km/s) to modify the spectra with')
    parser.add_argument('-vrot', '--rotational_vel', type=float, default=70.,
                        help='maximum rotational velocity (km/s) to modify the spectra with')
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
    parser.add_argument('--max_afe', type=float, default=np.Inf,
                        help='maximum [alpha/Fe]')
    parser.add_argument('--min_afe', type=float, default=-np.Inf,
                        help='minimum [alpha/Fe]')
    parser.add_argument('--sigma_gaussian', type=float, default=50.,
                        help='sigma of Gaussian kernel if using Gaussian smoothing'
                             'continuum normalization')

    # Collect arguments
    return parser.parse_args()


def main():

    # Collect arguments
    args = parse_args()

    # Check if supplied grid name is valid
    grid_name = args.grid.lower()
    if grid_name not in ['intrigoss', 'phoenix', 'ambre', 'ferre', 'nlte', 'mpia']:
        raise ValueError('{} not a valid grid name. Need to supply an appropriate spectral grid name '
                         '(phoenix, intrigoss, ferre, mpia, nlte, or ambre)'.format(grid_name))

    # Collect list of files containing the spectra
    file_list = collect_file_list(grid_name, args.spec_dir)

    # Get observational wavelength grid (stored in saved numpy array)
    wave_grid_obs = np.load(args.obs_wave_file)

    # Determine total number of spectra already in chosen directory
    existing_files = glob.glob(args.save_dir + '/*')
    # Number of spectra appended on to end of filename
    total_num_spec = np.sum([int(os.path.basename(f)[37:]) for f in existing_files])
    print('Total # of spectra in directory: {}/{}'.format(total_num_spec, args.max_num_spec))
    print('Number of spectra already processed: {}/{}'.format(total_num_spec, args.max_num_spec))

    # Generate the batches of spectra
    spectra_created = 0
    while total_num_spec <= args.max_num_spec:

        if total_num_spec >= args.max_num_spec:
            print('Maximum number of spectra reached.')
            break

        # Process a batch of raw synthetic spectra
        t0_batch = time.time()
        print('Creating a batch of spectra...')
        if args.telescope.lower() == 'aat':
            spec_asym, params = preprocess_batch_of_aat_spectra(file_list, wave_grid_obs,
                                                                batch_size=args.batch_size,
                                                                max_vrot_to_apply=args.rotational_vel,
                                                                max_vrad_to_apply=args.radial_vel,
                                                                max_noise=args.noise,
                                                                spectral_grid_name=grid_name,
                                                                synth_wave_filename=args.synth_wave_file,
                                                                max_teff=args.max_teff,
                                                                min_teff=args.min_teff,
                                                                max_logg=args.max_logg,
                                                                min_logg=args.min_logg,
                                                                max_feh=args.max_feh,
                                                                min_feh=args.min_feh)
        else:
            spec_asym, params = preprocess_batch_of_spectra(file_list, wave_grid_obs, args.resolution,
                                                            batch_size=args.batch_size,
                                                            max_vrot_to_apply=args.rotational_vel,
                                                            max_vrad_to_apply=args.radial_vel,
                                                            max_noise=args.noise,
                                                            spectral_grid_name=grid_name,
                                                            synth_wave_filename=args.synth_wave_file,
                                                            max_teff=args.max_teff,
                                                            min_teff=args.min_teff,
                                                            max_logg=args.max_logg,
                                                            min_logg=args.min_logg,
                                                            max_feh=args.max_feh,
                                                            min_feh=args.min_feh)
        teff, logg, m_h, a_m, vt, vrot, vrad, noise = params

        if np.shape(spec_asym)[0] == 0:
            print('there was an error... preprocessing next batch')
            continue

        # Save this batch to an h5 file in chosen save directory
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        unique_filename = str(uuid.uuid4()) + '_{}'.format(np.shape(spec_asym)[0])
        save_path = os.path.join(args.save_dir, unique_filename)
        print('Saving {} to {}'.format(unique_filename, args.save_dir))
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('spectra_asymnorm', data=np.asarray(spec_asym))
            f.create_dataset('teff', data=np.asarray(teff))
            f.create_dataset('logg', data=np.asarray(logg))
            f.create_dataset('M_H', data=np.asarray(m_h))
            f.create_dataset('a_M', data=np.asarray(a_m))
            f.create_dataset('VT', data=np.asarray(vt))
            f.create_dataset('v_rot', data=np.asarray(vrot))
            f.create_dataset('v_rad', data=np.asarray(vrad))
            f.create_dataset('noise', data=np.asarray(noise))
            f.create_dataset('wave_grid', data=np.asarray(wave_grid_obs))
        print('Total time to make this batch of {} spectra: {:.1f}'.format(np.shape(spec_asym)[0],
                                                                           time.time() - t0_batch))

        spectra_created += args.batch_size
        print('Total # of spectra created so far: {}'.format(spectra_created))

        # Again, check to see how many files are in the chosen save directory (parallel jobs will be filling it up too)
        existing_files = glob.glob(args.save_dir + '/*')
        # Number of spectra appended on to end of filename
        total_num_spec = np.sum([int(os.path.basename(f)[37:]) for f in existing_files])
        print('Total # of spectra in directory: {}/{}'.format(total_num_spec, args.max_num_spec))


if __name__ == '__main__':
    main()
