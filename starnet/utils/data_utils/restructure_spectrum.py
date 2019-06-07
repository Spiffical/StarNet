import astropy
import numpy as np
import multiprocessing
from astropy.stats import sigma_clip
from pysynphot import observation
from pysynphot import spectrum as pysynspec
from spectres import spectres
from contextlib import contextmanager
from scipy import interpolate

from starnet.utils.data_utils.augment import get_noise_of_segments


def rebin(new_wav, old_wav, flux):

    f_ = np.ones(len(old_wav))
    spec_ = pysynspec.ArraySourceSpectrum(wave=old_wav, flux=flux)
    filt = pysynspec.ArraySpectralElement(old_wav, f_, waveunits='angstrom')
    obs = observation.Observation(spec_, filt, binset=new_wav, force='taper')
    newflux = obs.binflux
    return newflux


def c_sigmaclip1D(flux, rms_noise, sigma_clip_threshold=2.0):
    """
    Perform corrected sigma-clipping (https://github.com/radio-astro-tools/statcont)
    to determine the mean flux level, with different adaptations for emission-
    and absorption-dominated spectra.
    It runs on one-dimensional arrays

    Parameters
    ----------
    flux : np.ndarray
        One-dimension array of flux values
    rms_noise : float
        The estimated RMS noise level of the data
    sigma_clip_threshold : float
        The threshold in number of sigma above/below which to reject outlier
        data

    Returns
    -------
    sigmaclip_flux : float
    sigmaclip_noise : float
        The measured continuum flux and estimated 1-sigma per-channel noise
        around that measurement
    """

    flux_copy = np.copy(flux)
    # Sigma-clipping method applied to the flux array
    if astropy.version.major >= 1:
        filtered_data = sigma_clip(flux_copy, sigma=sigma_clip_threshold,
                                   iters=None)
    elif astropy.version.major < 1:
        filtered_data = sigma_clip(flux_copy, sig=sigma_clip_threshold, iters=None)

    sigmaclip_flux_prev = sigmaclip_flux = np.mean(filtered_data)
    sigmaclip_noise = sigmaclip_sigma = np.std(filtered_data)
    mean_flux = np.mean(flux)

    # Correction of sigma-clip continuum level, making use of the
    # presence of emission and/or absorption line features

    # Set up the fraction of channels (in %) that are in emission
    fraction_emission = 0
    fraction_emission = sum(i > (sigmaclip_flux + 1 * rms_noise) for i in flux)
    fraction_emission = 100 * fraction_emission / len(flux)

    # Set up the fraction of channels (in %) that are in absorption
    fraction_absorption = 0
    fraction_absorption = sum(i < (sigmaclip_flux - 1 * rms_noise) for i in flux)
    fraction_absorption = 100 * fraction_absorption / len(flux)

    # Apply correction to continuum level
    # see details in Sect. 2.4 of Sanchez-Monge et al. (2017)
    if (fraction_emission < 33 and fraction_absorption < 33):
        sigmaclip_flux = sigmaclip_flux_prev
    elif (fraction_emission >= 33 and fraction_absorption < 33):
        if (fraction_emission - fraction_absorption > 25):
            sigmaclip_flux = sigmaclip_flux_prev - 1.0 * sigmaclip_sigma
        if (fraction_emission - fraction_absorption <= 25):
            sigmaclip_flux = sigmaclip_flux_prev - 0.5 * sigmaclip_sigma
    elif (fraction_emission < 33 and fraction_absorption >= 33):
        if (fraction_absorption - fraction_emission > 25):
            sigmaclip_flux = sigmaclip_flux_prev + 1.0 * sigmaclip_sigma
        if (fraction_absorption - fraction_emission <= 25):
            sigmaclip_flux = sigmaclip_flux_prev + 0.5 * sigmaclip_sigma
    elif (fraction_emission >= 33 and fraction_absorption >= 33):
        if (fraction_emission - fraction_absorption > 25):
            sigmaclip_flux = sigmaclip_flux_prev - 1.0 * sigmaclip_sigma
        if (fraction_absorption - fraction_emission > 25):
            sigmaclip_flux = sigmaclip_flux_prev + 1.0 * sigmaclip_sigma
        if (abs(fraction_absorption - fraction_emission) <= 25):
            sigmaclip_flux = sigmaclip_flux_prev

    return sigmaclip_flux


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


def fit_continuum(flux, line_regions, wav, segments_step=10, sigma_upper=2.0,
                  sigma_lower=0.5, fit='asym_sigmaclip', j=0):
    flux_copy = np.copy(flux)

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
            flux_copy[mask] = np.nan

    # Get rms noise if required
    if fit == 'c_sigmaclip':
        rms_noise = get_noise_of_segments(flux_copy, segments)

    # Determine continuum level in each segment
    for k in range(len(segments) - 1):
        flux_segment = flux_copy[segments[k]:segments[k + 1]]
        if sum(np.isnan(flux_segment)) > 0.5 * len(flux_segment):
            continue
        if fit == 'asym_sigmaclip':
            cont_ = asymmetric_sigmaclip1D(flux=flux_segment, sigma_upper=sigma_upper,
                                           sigma_lower=sigma_lower)
        elif fit == 'c_sigmaclip':
            cont_ = c_sigmaclip1D(flux=flux_segment, rms_noise=rms_noise)
        else:
            raise Exception('No recognized continuum fitting procedure')
        cont_array[segments[k]:segments[k + 1]] = cont_

    # Fill in NaNs and zeros with interpolated values
    prev_val = cont_array[0]
    beg_idx = 0
    for i, val in enumerate(cont_array):
        if (np.isnan(val) and not np.isnan(prev_val)) or (val == 0 and prev_val != 0):
            beg_idx = i - 1
            beg_val = prev_val
        elif (not np.isnan(val) and np.isnan(prev_val)) or (val != 0 and prev_val == 0):
            end_idx = i
            end_val = val

            # Interpolate between values on either side of the segment of NaNs/zeros
            x = [wav[beg_idx], wav[end_idx]]
            y = [beg_val, end_val]
            xvals = wav[beg_idx:end_idx]
            yinterp = np.interp(xvals, x, y)

            # Patch back into cont_array
            cont_array[beg_idx:end_idx] = yinterp
        prev_val = val

    # Fit a spline to the found continuum
    t = np.linspace(wav[0], wav[-1], 20)
    tck = interpolate.splrep(wav, cont_array, t=t[1:-1])
    cont_array = interpolate.splev(wav, tck, der=0)

    return cont_array, j


def continuum_normalize(flux, line_regions, wav, segments_step=10, fit='asym_sigmaclip', sigma_upper=2.0,
                        sigma_lower=0.5):
    flux_copy = np.copy(flux)
    cont, _ = fit_continuum(flux_copy, line_regions, wav, segments_step, sigma_upper, sigma_lower, fit=fit)
    norm_flux = flux_copy / cont

    return norm_flux, cont


def continuum_normalize_parallel(spectra, line_regions, wav, segments_step=10, fit='asym_sigmaclip', sigma_upper=2.0,
                                 sigma_lower=0.5):
    @contextmanager
    def poolcontext(*args, **kwargs):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()

    num_spec = np.shape(spectra)[0]
    num_cpu = multiprocessing.cpu_count()
    pool_size = num_cpu if num_spec >= num_cpu else num_spec

    pool_arg_list = [(spectra[i], line_regions, wav,
                      segments_step, fit, sigma_upper,
                      sigma_lower) for i in range(num_spec)]
    with poolcontext(processes=pool_size) as pool:
        results = pool.starmap(continuum_normalize, pool_arg_list)

    norm_fluxes = [result[0] for result in results]

    return norm_fluxes
