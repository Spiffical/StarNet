import os
import sys
import numpy as np
import random
import numpy.ma as ma
import scipy
import time
import multiprocessing
from astropy.stats import sigma_clip
from pysynphot import observation
from pysynphot import spectrum as pysynspec
from scipy import interpolate
from contextlib import contextmanager
from eniric.broaden import convolution

sys.path.insert(0, os.path.join(os.getenv('HOME'), 'StarNet'))
from starnet.utils.data_utils.loading import get_synth_wavegrid, get_synth_spec_data

# Define parameters needed for continuum fitting
LINE_REGIONS = [[3910, 3955], [3957, 3981], [4210, 4240], [4250, 4410], [4333, 4388], [4845, 4886], [5160, 5200], [5874, 5916], [6530, 6590],
                [8490, 8510], [8530, 8555], [8654, 8675]]
SEGMENTS_STEP = 10.  # divide the spectrum into segments of 10 Angstroms


def get_noise(flux):
    
    flux_copy = np.copy(flux)
        
    # Filter just once to take care of absorption lines
    filtered_data = sigma_clip(flux_copy, sigma_lower=0.5, sigma_upper=5.0,
                               iters=1)
    noise = np.std(filtered_data)
        
    return noise


def get_noise_of_segments(flux, segments):
    
    flux_copy = np.copy(flux)
    
    noise_array = []
    for k in range(len(segments)-1):
        flux_segment = flux_copy[segments[k]:segments[k+1]]
        noise_array.append(get_noise(flux_segment))
        
    return np.mean(noise_array)


def add_noise(x, noise=0.07):

    if type(noise) == float or type(noise) == int or type(noise) == np.float64:
        noise_factor = noise*np.median(x)
        x += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    else:
        raise ValueError('Noise parameter must be a float or integer')
    return x


def add_zeros(x, max_zeros=150):
    
    x = np.atleast_1d(x)
    for x_ in x:
        num_zeros_init = np.sum(x_==0)
        if num_zeros_init > max_zeros:
            continue
        else:
            num_zeros = random.randint(0, max_zeros - num_zeros_init)
            indices = random.sample(range(len(x_)), num_zeros)
            if len(indices)!=0:
                x_[indices] = 0
    return x


def add_radial_velocity(wav, rv, flux=None):
    """
    This function adds radial velocity effects to a spectrum.
    
    wav: wavelength array
    rv: radial velocity (km/s)
    flux: spectral flux array
    """
    # Speed of light in m/s
    c = 299792458.0

    # New wavelength array with added radial velocity
    new_wav = wav * np.sqrt((1.-(-rv*1000.)/c)/(1.+(-rv*1000.)/c))

    # if flux array provided, interpolate it onto this new wavelength grid and return both,
    # otherwise just return the new wavelength grid
    if flux is not None:
        t = time.time()
        new_flux = rebin(new_wav, wav, flux)
        print('[INFO] Time taken to rebin RV: {:.2f}s'.format(time.time() - t))
        return new_wav, new_flux
    else:
        return new_wav


def add_zeros_global_error(x, error_indx):
    
    for item in x:
        item[error_indx] = 0
    return x


def augment_spectrum(flux, wav, new_wav, rot=65, noise=0.02, vrad=200., to_res=20000):
    """

    :param flux: an array of flux values
    :param wav: synthetic wavelength grid
    :param new_wav: wavelength grid to rebin the synthetic wavelength grid to
    :param rot: value of rotational velocity (km/s) to apply to flux array
    :param noise: value of noise (fraction of flux value) to apply to flux array
    :param vrad: value of radial velocity (km/s) to apply to flux array
    :param to_res: resolution of output flux array requested

    :return: modified flux array
    """

    # Degrade resolution and apply rotational broadening
    epsilon = random.uniform(0, 1.)
    _, _, flux = convolution(wav=wav,
                             flux=flux,
                             vsini=rot,
                             R=to_res,
                             epsilon=epsilon,
                             normalize=True,
                             num_procs=10)

    # Add radial velocity
    if vrad != 0:
        rv_wav = add_radial_velocity(wav, vrad)
        wav = rv_wav

    # Rebin to new wave grid
    flux = rebin(new_wav, wav, flux)

    # Add noise
    flux = add_noise(flux, noise)

    return flux


def augment_spectra_parallel(spectra, wav, new_wav, vrot_list, noise_list, vrad_list, instrument_res, ):
    """
    Augments (in parallel) a list of spectra with rotational velocity, radial velocity, noise, and resolution
    degradation.

    :param spectra: list of spectra
    :param wav: list of synthetic wavelength grids
    :param new_wav: wavelength grid to rebin the synthetic wavelength grid to
    :param vrot_list: a list, same length as spectra list, of rotational velocities (km/s) to apply
    :param noise_list: a list, same length as spectra list, of maximum noise (fraction of flux) to apply
    :param vrad_list: a list, same length as spectra list, of radial velocities (km/s) to apply
    :param instrument_res: instrumental resolution to degrade the synthetic spectra to

    :return: a list of modified input spectra
    """

    @contextmanager
    def poolcontext(*args, **kwargs):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()

    num_spectra = np.shape(spectra)[0]
    num_cpu = multiprocessing.cpu_count()
    pool_size = num_cpu if num_spectra >= num_cpu else num_spectra
    print('[INFO] Pool size: {}'.format(pool_size))

    # Degrade resolution and apply rotational broadening
    # for i in range(len(spectra)):
    #     dw = wav[i][1] - wav[i][0]
    #     dw_new = new_wav[1] - new_wav[0]
    #
    #     dw_intermed = (dw + dw_new) / 2.
    #     num_wvl_intermed = int((wav[i][-1] - wav[i][0]) / dw_intermed)
    #     wav_intermed = np.asarray([wav[i][0] + dw_intermed*j for j in range(num_wvl_intermed)])
    #
    #     print('[INTERMEDIATE STEP] dw old: {}'.format(dw))
    #     print('[INTERMEDIATE STEP] dw new: {}'.format(dw_new))
    #     print('[INTERMEDIATE STEP] dw intermediate: {}'.format(dw_intermed))
    #     print('[INTERMEDIATE STEP] shapes old/intermed: {}/{}'.format(np.shape(wav[i]), np.shape(wav_intermed)))
    #
    #     flux = rebin(wav_intermed, wav[i], spectra[i])
    #     wav[i] = wav_intermed
    #     spectra[i] = flux
    #
    #     epsilon = random.uniform(0, 1.)
    #     _, _, flux = convolution(wav=wav[i],
    #                              flux=spectra[i],
    #                              vsini=vrot_list[i],
    #                              R=instrument_res,
    #                              epsilon=epsilon,
    #                              normalize=True,
    #                              num_procs=10)
    #     spectra[i] = flux

    pool_arg_list = [(spectra[i], wav[i], new_wav, vrot_list[i], noise_list[i], vrad_list[i], instrument_res)
                     for i in range(num_spectra)]
    with poolcontext(processes=pool_size) as pool:
        results = pool.starmap(augment_spectrum, pool_arg_list)

    return results


def telluric_mask(telluric_line_file, wav):
    # Load .txt file containing information about telluric lines
    telluric_lines = np.loadtxt(telluric_line_file, skiprows=1)

    # Extract relevant information
    telluric_regions = np.column_stack((telluric_lines[:, 0], telluric_lines[:, 1]))
    #residual_intensity = telluric_lines[:,2]

    # Generate telluric mask
    telluric_mask = np.ones(len(wav))
    for region in telluric_regions:
        lower_wl, upper_wl = region
        mask = (wav>lower_wl) & (wav<upper_wl)
        telluric_mask[mask] = 0
        
    return telluric_mask


def mask_tellurics(telluric_line_file, X, wav):

    mask = telluric_mask(telluric_line_file, wav)
    
    if np.ndim(X) == 1:
        X *= mask
    elif np.ndim(X) == 2:
        for x in X:
            x *= mask
    
    return X


def rebin(new_wav, old_wav, flux):

    f_ = np.ones(len(old_wav))
    spec_ = pysynspec.ArraySourceSpectrum(wave=old_wav, flux=flux)
    filt = pysynspec.ArraySpectralElement(old_wav, f_, waveunits='angstrom')
    obs = observation.Observation(spec_, filt, binset=new_wav, force='taper')
    newflux = obs.binflux
    return newflux


def asymmetric_sigmaclip1D(flux, sigma_upper=2.0, sigma_lower=0.5):
    """
    Perform asymmetric sigma-clipping to determine the mean flux level
    It runs on one-dimensional arrays
    Parameters
    ----------
    flux : np.ndarray
        One-dimension array of flux values
    sigma_lower : float
        The threshold in number of sigma below which to reject outlier
        data
    sigma_upper : float
        The threshold in number of sigma above which to reject outlier
        data
    Returns
    -------
    sigmaclip_flux : float
        The measured continuum flux
    """

    flux_copy = np.copy(flux)

    # Iteratively sigma clip the flux array
    filtered_data = sigma_clip(flux_copy, sigma_lower=sigma_lower, sigma_upper=sigma_upper,
                               iters=None)

    sigmaclip_flux = np.mean(filtered_data)

    return sigmaclip_flux


def gaussian_smooth_continuum(flux, wave_grid, err=None, sigma=50, sigma_cutoff=4):

    # Mask the flux array according to the error array
    if err is not None:
        flux = ma.array(flux, mask=err)
    else:
        flux = ma.array(flux, mask=np.zeros(len(flux)))

    # Calculate the Gaussian kernel with a given sigma (in Angstroms), cutting it off at sigma_cutoff*sigma
    dx = wave_grid[1] - wave_grid[0]
    gx = np.arange(-sigma_cutoff * sigma, sigma_cutoff * sigma, dx)
    if len(gx) % 2 == 0:
        gx = np.arange(-sigma_cutoff * sigma, sigma_cutoff * sigma + dx, dx)
    gaussian = np.exp(-(gx / sigma) ** 2)

    # Determine the sum of the Gaussian kernel when centered on different points along the flux array (only
    # taking into account the overlapping regions)
    sum_gaussian = np.sum(gaussian)
    sums = []
    if len(gaussian) < len(flux):
        for i in range(int(len(gaussian) / 2)):
            sums.append(np.sum(gaussian[int(len(gaussian) / 2) - i:]))
        for i in range(len(flux) - len(gaussian)):
            sums.append(sum_gaussian)
        for i in range(int(len(gaussian) / 2) + 1):
            sums.append(np.sum(gaussian[:len(gaussian) - i]))
    else:
        # TODO
        raise Exception('Gaussian kernel longer than flux array, this is not implemented yet. Choose a'
                        ' smaller sigma or smaller sigma cutoff')
        # for i in range(len(flux)):
        #    sums.append(gaussian[])

    # Convolve the Gaussian with the flux array, normalizing each point
    result = np.convolve(flux.filled(0), gaussian, mode="same") / sums

    return result


def gaussian_smooth_continuum_old(flux, wave_grid, err=None):

    W = 50  # sigma of 50 Angstroms for the Gaussian kernel
    cont = []
    for i, w in enumerate(wave_grid):
        dist = wave_grid - w
        K = scipy.exp(-dist ** 2 / W ** 2)
        if err is not None:
            good_indices = (K != 0) & (err != 1)
        else:
            good_indices = K != 0
        cont_val = np.dot(flux[good_indices], K[good_indices]) / np.sum(K[good_indices])
        cont.append(cont_val)
    return cont


def fit_continuum(flux, wav, err=None, line_regions=None, segments_step=10, sigma_upper=2.0,
                  sigma_lower=0.5, sigma_gaussian=50, fit='asym_sigmaclip', spline_fit=20):
    flux_copy = np.copy(flux)

    if fit.lower() == 'gaussian_smooth':
        cont_array = gaussian_smooth_continuum(flux_copy, wav, err, sigma=sigma_gaussian)
    elif fit.lower() in ['asym_sigmaclip', 'c_sigmaclip']:

        masked_flux = ma.masked_array(flux_copy, mask=flux_copy == 0)

        # Create an array of indices that segment the flux array in steps of segments_step [Angstroms]
        segments = range(0, len(wav), int(segments_step / (wav[1] - wav[0])))
        segments = list(segments)
        segments.append(len(wav))

        # Initialize continuum array
        cont_array = np.empty(len(wav))
        cont_array[:] = np.nan

        # Mask line regions
        if line_regions is not None:
            for lower_wl, upper_wl in line_regions:
                mask = (wav > lower_wl) & (wav < upper_wl)
                masked_flux[mask] = np.nan

        # Determine continuum level in each segment
        for k in range(len(segments) - 1):
            flux_segment = masked_flux[segments[k]:segments[k + 1]]
            if sum(np.isnan(flux_segment)) > (0.5 * len(flux_segment)):
                continue
            if sum(flux_segment == 0) > (0.5 * len(flux_segment)):
                continue
            cont_ = asymmetric_sigmaclip1D(flux=flux_segment, sigma_upper=sigma_upper,
                                           sigma_lower=sigma_lower)
            cont_array[segments[k]:segments[k + 1]] = cont_

        # Fill continuum array with zero values where needed so spline fitting ignores them
        cont_array[flux == 0] = 0
        cont_array[np.isnan(flux)] = 0
        cont_array[np.isnan(cont_array)] = 0
        if line_regions is not None:
            for lower_wl, upper_wl in line_regions:
                mask = (wav > lower_wl) & (wav < upper_wl)
                cont_array[mask] = 0

        # Find the indices for where leading/trailing zeros end/begin
        m = cont_array != 0
        leading_indx, trailing_indx = m.argmax() - 1, m.size - m[::-1].argmax()
        leading_indx += 1

        # Fit a spline to the found continuum (ignoring the leading/trailing zeros)
        t = np.linspace(wav[leading_indx], wav[trailing_indx], spline_fit)
        w = np.ones(len(wav))[leading_indx:trailing_indx]
        w[cont_array[leading_indx:trailing_indx] == 0] = 0
        tck = interpolate.splrep(wav[leading_indx:trailing_indx],
                                 cont_array[leading_indx:trailing_indx],
                                 t=t[1:-1],
                                 w=w)
        cont_array = interpolate.splev(wav[leading_indx:trailing_indx], tck, der=0)

        # Add the leading/trailing zeros back in
        cont_array = np.pad(cont_array, (leading_indx, len(flux) - trailing_indx), 'constant')

        # NaN mask the continuum so we don't get bad values when dividing it into the flux
        cont_array[cont_array < 0] = np.nan
        cont_array[flux == 0] = np.nan
    else:
        raise ValueError('{} continuum method not recognized'.format(fit))

    return cont_array


def continuum_normalize(flux, wav, err=None, line_regions=None, segments_step=10,
                        sigma_upper=2.0, sigma_lower=0.5, sigma_gaussian=50, fit='asym_sigmaclip', spline_fit=20):

    if flux is None:
        return None
    else:
        flux_copy = np.copy(flux)

        if len(np.shape(flux_copy)) == 1:
            cont = fit_continuum(flux_copy, wav, err, line_regions, segments_step,
                                 sigma_upper, sigma_lower, sigma_gaussian, fit, spline_fit)
            norm_flux = flux_copy / cont
        else:
            norm_flux = []
            for i in range(np.shape(flux)[0]):
                if err is not None:
                    cont_ = fit_continuum(flux_copy[i], wav, err[i], line_regions, segments_step,
                                          sigma_upper, sigma_lower, sigma_gaussian, fit, spline_fit)
                else:
                    cont_ = fit_continuum(flux_copy[i], wav, err, line_regions, segments_step,
                                          sigma_upper, sigma_lower, sigma_gaussian, fit, spline_fit)
                norm_flux.append(flux_copy[i] / cont_)
            norm_flux = np.asarray(norm_flux)

        return norm_flux


def continuum_normalize_parallel(spectra, wav, err=None, line_regions=None, segments_step=10.,
                                 sigma_upper=2.0, sigma_lower=0.5, sigma_gaussian=50, fit='asym_sigmaclip'):
    @contextmanager
    def poolcontext(*args, **kwargs):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()

    num_spec = np.shape(spectra)[0]
    num_cpu = multiprocessing.cpu_count()
    pool_size = num_cpu if num_spec >= num_cpu else num_spec

    pool_arg_list = [(spectra[i], wav, err, line_regions,
                      segments_step, sigma_upper,
                      sigma_lower, sigma_gaussian, fit) for i in range(num_spec)]
    with poolcontext(processes=pool_size) as pool:
        results = pool.starmap(continuum_normalize, pool_arg_list)

    norm_fluxes = [result for result in results]

    return norm_fluxes


def collect_batch(file_list, wave_grid_obs, spectral_grid_name='phoenix', synth_wave_filename=None,
                   max_teff=np.Inf,  min_teff=-np.Inf, max_logg=np.Inf, min_logg=-np.Inf, max_feh=np.Inf,
                   min_feh=-np.Inf, max_afe=np.Inf, min_afe=-np.Inf):

    # Based on the observed wavelength grid, define a wavelength range to slice the synthetic wavelength grid
    # and spectrum (for quicker processing times). Extend on both sides to accommodate radial velocity shifts
    # on synthetic spectra, and also to eliminate edge effects from rotational broadening
    extension = 5  # Angstroms
    wave_min_request = wave_grid_obs[0] - extension
    wave_max_request = wave_grid_obs[-1] + extension

    # Initialize lists
    spectra = []
    wavegrid_synth_list = []
    teff_list = []
    logg_list = []
    vt_list = []
    m_h_list = []
    a_m_list = []
    batch_file_list = []

    # This next block of code will iteratively select a random file from the supplied list of synthetic spectra files,
    # extract the flux and stellar parameters from it, and if it falls within the requested parameter space, will
    # append the data to the list of spectra to be modified. It will repeat until the requested batch size has been met.
    for batch_file in file_list:

        # Collect flux and stellar parameters from file
        flux, wave_grid_synth, params = get_synth_spec_data(batch_file, spectral_grid_name)
        teff = params['teff']
        logg = params['logg']
        m_h = params['m_h']
        a_m = params['a_m']
        vt = params['vt']
        #vrot = params['vrot']

        # Skip if flux is mostly NaNs
        if sum(np.isnan(flux)) > 0.5 * len(flux):
            print('Spectrum has too many NaNs! Skipping...')
            print('>>> Spectrum length: {}'.format(len(flux)))
            print('>>> Number of NaNs:  {}'.format(sum(np.isnan(flux))))
            print('>>> Parameters: {} {} {} {}'.format(teff, logg, m_h, a_m))
            continue

        # Skip this spectrum if beyond requested temperature, logg, or metallicity limits
        if (teff > max_teff) or (teff < min_teff) or (logg > max_logg) or (logg < min_logg) or \
                (m_h > max_feh) or (m_h < min_feh) or (a_m > max_afe) or (a_m < min_afe):
            continue
        else:
            # Get synthetic wavelength grid
            #if spectral_grid_name == 'intrigoss' or spectral_grid_name == 'ambre' or spectral_grid_name == 'ferre' \
            #        or spectral_grid_name == 'nlte' or spectral_grid_name == 'mpia':
            #    synth_wave_filename = batch_file
            if spectral_grid_name == 'phoenix':
                if synth_wave_filename is None or synth_wave_filename.lower() == 'none':
                    raise ValueError('for Phoenix grid, need to supply separate file containing wavelength grid')
                else:
                    wave_grid_synth = get_synth_wavegrid(synth_wave_filename, spectral_grid_name)
            #else:
            #    synth_wave_filename = None
            #wave_grid_synth = get_synth_wavegrid(synth_wave_filename, spectral_grid_name)

            # Trim wave grid and flux array
            wave_indices = (wave_grid_synth > wave_min_request) & (wave_grid_synth < wave_max_request)
            wave_grid_synth = wave_grid_synth[wave_indices]
            flux = flux[wave_indices]

            # Check for repeating wavelengths and remove them
            dw = wave_grid_synth[1:] - wave_grid_synth[:-1]
            idx = np.where(dw == 0)[0]
            if len(idx) > 0:
                wave_grid_synth = np.delete(wave_grid_synth, idx)
                flux = np.delete(flux, idx)
                print('DELETING REPEATING WVL ENTRIES: {}'.format(idx))

            # Fill up lists
            spectra.append(flux)
            wavegrid_synth_list.append(wave_grid_synth)
            teff_list.append(teff)
            logg_list.append(logg)
            m_h_list.append(m_h)
            a_m_list.append(a_m)
            vt_list.append(vt)

            batch_file_list.append(batch_file)

    return spectra, wavegrid_synth_list, teff_list, logg_list, m_h_list, a_m_list, vt_list, batch_file_list


def preprocess_batch_of_spectra(file_list, wave_grid_obs, instrument_res, batch_size=32, max_vrot_to_apply=70,
                   max_vrad_to_apply=200, max_noise=0.07, spectral_grid_name='phoenix', synth_wave_filename=None,
                   max_teff=np.Inf,  min_teff=-np.Inf, max_logg=np.Inf, min_logg=-np.Inf, max_feh=np.Inf,
                   min_feh=-np.Inf, max_afe=np.Inf, min_afe=-np.Inf):


    # This next block of code will iteratively select a random file from the supplied list of synthetic spectra files,
    # extract the flux and stellar parameters from it, and if it falls within the requested parameter space, will
    # append the data to the list of spectra to be modified. It will repeat until the requested batch size has been met.
    t1 = time.time()

    # Randomly choose file names from the whole list of synthetic spectra files
    batch_indices = np.random.choice(np.shape(file_list)[0], batch_size, replace=False)
    batch_filenames = np.asarray(file_list)[batch_indices]

    spectra, wavegrid_synth_list, teff_list, logg_list, m_h_list, a_m_list, vt_list, \
    batch_file_list = collect_batch(batch_filenames, wave_grid_obs,
                                    spectral_grid_name,
                                    synth_wave_filename,
                                    max_teff, min_teff,
                                    max_logg, min_logg,
                                    max_feh, min_feh,
                                    max_afe, min_afe)

    print('Time taken to collect spectra: %.1f s' % (time.time() - t1))
    print('File paths for collected spectra: {}'.format(batch_file_list))

    vrad_list = np.random.uniform(low=-max_vrad_to_apply, high=max_vrad_to_apply, size=(batch_size,))
    vrot_list = np.random.uniform(low=0, high=max_vrot_to_apply, size=(batch_size,))
    noise_list = np.random.uniform(low=0, high=max_noise, size=(batch_size,))

    t1 = time.time()
    # Modify spectra in parallel (degrade resolution, apply rotational broadening, etc.)
    spectra = augment_spectra_parallel(spectra, wavegrid_synth_list, wave_grid_obs, vrot_list, noise_list,
                                       vrad_list, instrument_res)
    print('Total modify time: %.1f s' % (time.time() - t1))

    # Continuum normalize spectra with asymmetric sigma clipping continuum fitting method
    t1 = time.time()
    spectra = continuum_normalize_parallel(spectra, wave_grid_obs,
                                           line_regions=LINE_REGIONS,
                                           segments_step=SEGMENTS_STEP,
                                           fit='asym_sigmaclip',
                                           sigma_upper=2.0, sigma_lower=0.5)
    print('Total continuum time: %.2f s' % (time.time() - t1))

    # Check again for nans
    nan_indices = []
    for i, f in enumerate(spectra):
        if sum(np.isnan(f)) > 0.5 * len(f):
            nan_indices.append(i)

    spectra = np.delete(spectra, nan_indices)
    teff_list = np.delete(teff_list, nan_indices)
    logg_list = np.delete(logg_list, nan_indices)
    m_h_list = np.delete(m_h_list, nan_indices)
    a_m_list = np.delete(a_m_list, nan_indices)
    vt_list = np.delete(vt_list, nan_indices)
    vrot_list = np.delete(vrot_list, nan_indices)
    vrad_list = np.delete(vrad_list, nan_indices)
    noise_list = np.delete(noise_list, nan_indices)

    params = teff_list, logg_list, m_h_list, a_m_list, vt_list, vrot_list, vrad_list, noise_list
    return spectra, params


def preprocess_batch_of_aat_spectra(file_list, wave_grid_obs, batch_size=32, max_vrot_to_apply=70,
                   max_vrad_to_apply=200, max_noise=0.07, spectral_grid_name='phoenix', synth_wave_filename=None,
                   max_teff=np.Inf,  min_teff=-np.Inf, max_logg=np.Inf, min_logg=-np.Inf, max_feh=np.Inf,
                   min_feh=-np.Inf, max_afe=np.Inf, min_afe=-np.Inf):

    # parse the wave grid to obtain the red and blue regions
    wave_grid_obs_blue = wave_grid_obs[:1854]
    wave_grid_obs_red = wave_grid_obs[1854:]

    # This next block of code will iteratively select a random file from the supplied list of synthetic spectra files,
    # extract the flux and stellar parameters from it, and if it falls within the requested parameter space, will
    # append the data to the list of spectra to be modified. It will repeat until the requested batch size has been met.
    t1 = time.time()

    # Randomly choose file names from the whole list of synthetic spectra files
    batch_indices = np.random.choice(np.shape(file_list)[0], batch_size, replace=False)
    batch_filenames = np.asarray(file_list)[batch_indices]

    spectra_blue, wavegrid_synth_list_blue, teff_list, logg_list, m_h_list, a_m_list, vt_list, \
    batch_file_list_blue = collect_batch(batch_filenames, wave_grid_obs_blue,
                                         spectral_grid_name,
                                         synth_wave_filename,
                                         max_teff, min_teff,
                                         max_logg, min_logg,
                                         max_feh, min_feh,
                                         max_afe, min_afe)

    spectra_red, wavegrid_synth_list_red, teff_list, logg_list, m_h_list, a_m_list, vt_list, \
    batch_file_list_red = collect_batch(batch_filenames, wave_grid_obs_red,
                                        spectral_grid_name,
                                        synth_wave_filename,
                                        max_teff, min_teff,
                                        max_logg, min_logg,
                                        max_feh, min_feh,
                                        max_afe, min_afe)

    t2 = time.time()
    print('Time taken to collect spectra: %.1f s' % (t2 - t1))
    print('File paths for collected spectra: {}'.format(batch_filenames))

    # Batch size may have changed after getting rid of some spectra
    batch_size = np.shape(spectra_red)[0]

    if batch_size > 0:
        vrad_list = np.random.uniform(low=-max_vrad_to_apply, high=max_vrad_to_apply, size=(batch_size,))
        vrot_list = np.random.uniform(low=0, high=max_vrot_to_apply, size=(batch_size,))
        noise_list = np.random.uniform(low=0, high=max_noise, size=(batch_size,))

        # Modify spectra in parallel (degrade resolution, apply rotational broadening, etc.)
        t1 = time.time()
        spectra_blue = augment_spectra_parallel(spectra_blue, wavegrid_synth_list_blue,
                                                wave_grid_obs_blue, vrot_list, noise_list, vrad_list,
                                                instrument_res=1300)
        spectra_red = augment_spectra_parallel(spectra_red, wavegrid_synth_list_red,
                                               wave_grid_obs_red, vrot_list, noise_list, vrad_list,
                                               instrument_res=11000)
        print('Total modify time: %.1f s' % (time.time() - t1))

        # Continuum normalize spectra with asymmetric sigma clipping continuum fitting method
        t1 = time.time()
        spectra_blue = continuum_normalize_parallel(spectra_blue, wave_grid_obs_blue,
                                                    line_regions=LINE_REGIONS,
                                                    segments_step=SEGMENTS_STEP,
                                                    fit='asym_sigmaclip',
                                                    sigma_upper=2.0, sigma_lower=0.5)
        spectra_red = continuum_normalize_parallel(spectra_red, wave_grid_obs_red,
                                                   line_regions=LINE_REGIONS,
                                                   segments_step=SEGMENTS_STEP,
                                                   fit='asym_sigmaclip',
                                                   sigma_upper=2.0, sigma_lower=0.5)
        print('Total continuum time: %.2f s' % (time.time() - t1))

        # Remove spectra with mostly NaNs
        nan_indices_blue = []
        for i, f in enumerate(spectra_blue):
            if sum(np.isnan(f)) > 0.5 * len(f):
                nan_indices_blue.append(i)
        nan_indices_red = []
        for i, f in enumerate(spectra_red):
            if sum(np.isnan(f)) > 0.5 * len(f):
                nan_indices_red.append(i)
        nan_indices = np.unique(np.append(nan_indices_blue, nan_indices_red))

        print('nan indices: {}'.format(nan_indices))

        if len(nan_indices) > 0:
            spectra_blue = np.delete(spectra_blue, nan_indices)
            spectra_red = np.delete(spectra_red, nan_indices)

        print('Shape of blue/red spectra: {}/{}'.format(np.shape(spectra_blue), np.shape(spectra_red)))
        spectra = np.concatenate((spectra_blue, spectra_red), axis=1)

        teff_list = np.delete(teff_list, nan_indices)
        logg_list = np.delete(logg_list, nan_indices)
        m_h_list = np.delete(m_h_list, nan_indices)
        a_m_list = np.delete(a_m_list, nan_indices)
        vt_list = np.delete(vt_list, nan_indices)
        vrot_list = np.delete(vrot_list, nan_indices)
        vrad_list = np.delete(vrad_list, nan_indices)
        noise_list = np.delete(noise_list, nan_indices)

        params = teff_list, logg_list, m_h_list, a_m_list, vt_list, vrot_list, vrad_list, noise_list
    else:
        spectra, params = [], []
    return spectra, params