import numpy as np
import random
import time
from astropy.stats import sigma_clip


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


def add_noise(x, noise):

    noise_factor = noise*np.median(x)
    x += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape) 
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
        from starnet.utils.data_utils.restructure_spectrum import rebin
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


def rebin_old(new_wav, old_wav, flux):

    # Ensure the new wavelength grid falls within the old wavelength grid (need for spectres resampling function,
    # and unfortunately the code requires that the endpoints of the wavelength arrays cannot even be identical, the
    # new wavelengths must have a larger minimum value and a smaller maximum value)
    lhs_new = new_wav[0]
    lhs_old = old_wav[0]
    rhs_new = new_wav[-1]
    rhs_old = old_wav[-1]

    if lhs_new <= lhs_old and rhs_new <= rhs_old:
        indices = new_wav >= lhs_old
        new_wav = new_wav[indices][1:-1]
        flux = flux[indices][1:-1]
    elif lhs_new >= lhs_old and rhs_new >= rhs_old:
        indices = new_wav <= rhs_old
        new_wav = new_wav[indices][1:-1]
        flux = flux[indices][1:-1]

    # Call the spectres function to resample the input spectrum or spectra to the new wavelength grid
    spec_resample = spectres(new_wav, old_wav, flux)

    return spec_resample


def convolve_spectrum(waveobs, flux, err, to_resolution, from_resolution=None):
    """
    Spectra resolution smoothness/degradation. Procedure:

    1) Define a bin per measure which marks the wavelength range that it covers.
    2) For each point, identify the window segment to convolve by using the bin widths and the FWHM.
    3) Build a gaussian using the sigma value and the wavelength values of the spectrum window.
    4) Convolve the spectrum window with the gaussian and save the convolved value.

    If "from_resolution" is not specified or its equal to "to_resolution", then the spectrum
    is convolved with the instrumental gaussian defined by "to_resolution".

    If "to_resolution" is specified, the convolution is made with the difference of
    both resolutions in order to degrade the spectrum.
    """
    if from_resolution is not None and from_resolution <= to_resolution:
        raise Exception("This method cannot deal with final resolutions that are bigger than original")

    total_points = len(waveobs)
    convolved_flux = np.zeros(total_points)
    convolved_err = np.zeros(total_points)

    # Consider the wavelength of the measurements as the center of the bins
    # Calculate the wavelength distance between the center of each bin
    wave_distance = waveobs[1:] - waveobs[:-1]
    # Define the edge of each bin as half the wavelength distance to the bin next to it
    edges_tmp = waveobs[:-1] + 0.5 * (wave_distance)
    # Define the edges for the first and last measure which where out of the previous calculations
    first_edge = waveobs[0] - 0.5*wave_distance[0]
    last_edge = waveobs[-1] + 0.5*wave_distance[-1]
    # Build the final edges array
    edges = np.array([first_edge] + edges_tmp.tolist() + [last_edge])

    # Bin width
    bin_width = edges[1:] - edges[:-1]          # width per pixel

    # FWHM of the gaussian for the given resolution
    if from_resolution is None:
        # Convolve using instrumental resolution (smooth but not degrade)
        fwhm = waveobs / to_resolution
    else:
        # Degrade resolution
        fwhm = __get_fwhm(waveobs, from_resolution, to_resolution)
    sigma = __fwhm_to_sigma(fwhm)
    # Convert from wavelength units to bins
    fwhm_bin = fwhm / bin_width

    # Round number of bins per FWHM
    nbins = np.ceil(fwhm_bin) #npixels

    # Number of measures
    nwaveobs = len(waveobs)

    # In theory, len(nbins) == len(waveobs)
    for i in np.arange(len(nbins)):
        current_nbins = 2 * nbins[i] # Each side
        current_center = waveobs[i] # Center
        current_sigma = sigma[i]

        # Find lower and uper index for the gaussian, taking care of the current spectrum limits
        lower_pos = int(max(0, i - current_nbins))
        upper_pos = int(min(nwaveobs, i + current_nbins + 1))

        # Select only the flux values for the segment that we are going to convolve
        flux_segment = flux[lower_pos:upper_pos+1]
        err_segment = err[lower_pos:upper_pos+1]
        waveobs_segment = waveobs[lower_pos:upper_pos+1]

        # Build the gaussian corresponding to the instrumental spread function
        gaussian = np.exp(- ((waveobs_segment - current_center)**2) / (2*current_sigma**2)) / np.sqrt(2*np.pi*current_sigma**2)
        gaussian = gaussian / np.sum(gaussian)

        # Convolve the current position by using the segment and the gaussian
        if flux[i] > 0:
            # Zero or negative values are considered as gaps in the spectrum
            only_positive_fluxes = flux_segment > 0
            weighted_flux = flux_segment[only_positive_fluxes] * gaussian[only_positive_fluxes]
            current_convolved_flux = weighted_flux.sum()
            convolved_flux[i] = current_convolved_flux
        else:
            convolved_err[i] = 0.0

        if err[i] > 0:
            # * Propagate error Only if the current value has a valid error value assigned
            #
            # Error propagation considering that measures are dependent (more conservative approach)
            # because it is common to find spectra with errors calculated from a SNR which
            # at the same time has been estimated from all the measurements in the same spectra
            #
            weighted_err = err_segment * gaussian
            current_convolved_err = weighted_err.sum()
            #current_convolved_err = np.sqrt(np.power(weighted_err, 2).sum()) # Case for independent errors
            convolved_err[i] = current_convolved_err
        else:
            convolved_err[i] = 0.0

    return waveobs, convolved_flux, convolved_err


def __fwhm_to_sigma(fwhm):
    """
    Calculate the sigma value from the FWHM.
    """
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    return sigma


class _Gdl:
    
    def __init__(self, vsini, epsilon):
        """
        Calculate the broadening profile.

        Parameters
        ----------
        vsini : float
        Projected rotation speed of the star [km/s]
        epsilon : float
        Linear limb-darkening coefficient
        """
        self.vc = vsini / 299792.458
        self.eps = epsilon
    
    def gdl(self, dl, refwvl, dwl):
        """
          Calculates the broadening profile.

          Parameters
          ----------
          dl : array
              'Delta wavelength': The distance to the reference point in
              wavelength space [A].
          refwvl : array
              The reference wavelength [A].
          dwl : float
              The wavelength bin size [A].

          Returns
          -------
          Broadening profile : array
              The broadening profile according to Gray. 
        """
        self.dlmax = self.vc * refwvl
        self.c1 = 2.*(1.- self.eps) / (np.pi * self.dlmax * (1. - self.eps/3.))
        self.c2 = self.eps / (2.* self.dlmax * (1. - self.eps/3.))
        result = np.zeros(len(dl))
        x = dl/self.dlmax
        indi = np.where(np.abs(x) < 1.0)[0]
        result[indi] = self.c1*np.sqrt(1. - x[indi]**2) + self.c2*(1. - x[indi]**2)
        # Correct the normalization for numeric accuracy
        # The integral of the function is normalized, however, especially in the case
        # of mild broadening (compared to the wavelength resolution), the discrete
        # broadening profile may no longer be normalized, which leads to a shift of
        # the output spectrum, if not accounted for.
        result /= (np.sum(result) * dwl)
        return result


def rotBroad(wvl, flux, epsilon, vsini, edgeHandling="firstlast"):
    """
    Apply rotational broadening to a spectrum.

    This function applies rotational broadening to a given
    spectrum using the formulae given in Gray's "The Observation
    and Analysis of Stellar Photospheres". It allows for
    limb darkening parameterized by the linear limb-darkening law.

    The `edgeHandling` parameter determines how the effects at
    the edges of the input spectrum are handled. If the default
    option, "firstlast", is used, the input spectrum is internally
    extended on both sides; on the blue edge of the spectrum, the
    first flux value is used and on the red edge, the last value
    is used to extend the flux array. The extension is neglected
    in the return array. If "None" is specified, no special care
    will be taken to handle edge effects.

    .. note:: Currently, the wavelength array as to be regularly
              spaced.

    Parameters
    ----------
    wvl : array
        The wavelength array [A]. Note that a
        regularly spaced array is required.
    flux : array
        The flux array.
    vsini : float
        Projected rotational velocity [km/s].
    epsilon : float
        Linear limb-darkening coefficient (0-1).
    edgeHandling : string, {"firstlast", "None"}
        The method used to handle edge effects.

    Returns
    -------
    Broadened spectrum : array
        An array of the same size as the input flux array,
        which contains the broadened spectrum.
    """
    
    if vsini <= 0.0:
        raise ValueError("vsini must be positive.")
    if (epsilon < 0) or (epsilon > 1.0):
        raise ValueError("Linear limb-darkening coefficient, epsilon, should be '0 < epsilon < 1'.")
    
    # Check whether wavelength array is evenly spaced
    sp = wvl[1::] - wvl[0:-1]
    sp = np.append(abs(wvl[0] - wvl[1]), sp)
    sp = sp.round(decimals=5)
    
    unique_vals, ind, counts = np.unique(sp, return_index=True, return_counts=True)
    
    if len(unique_vals)==1:  # Wavelength array is evenly spaced
    
        # Wavelength binsize
        dwl = wvl[1] - wvl[0]

        # Indices of the flux array to be returned
        validIndices = None

        if edgeHandling == "firstlast":
            # Number of bins additionally needed at the edges 
            binnu = int(np.floor(((vsini / 299792.458) * max(wvl)) / dwl)) + 1
            # Defined 'valid' indices to be returned
            validIndices = np.arange(len(flux)) + binnu
            # Adapt flux array
            front = np.ones(binnu) * flux[0]
            end = np.ones(binnu) * flux[-1]
            flux = np.concatenate( (front, flux, end) )
            # Adapt wavelength array
            front = (wvl[0] - (np.arange(binnu) + 1) * dwl)[::-1]
            end = wvl[-1] + (np.arange(binnu) + 1) * dwl
            wvl = np.concatenate( (front, wvl, end) )
        elif edgeHandling == "None":
            validIndices = np.arange(len(flux))
        else:
            raise ValueError("Edge handling method '" + str(edgeHandling) + "' currently not supported.")
        result = np.zeros(len(flux))
        gdl = _Gdl(vsini, epsilon)

        for i in range(len(flux)):
            dl = wvl[i] - wvl
            g = gdl.gdl(dl, wvl[i], dwl)
            result[i] = np.sum(flux * g)
        result *= dwl

        return result[validIndices]

    elif len(unique_vals) > 1:  # Wavelength array is sampled differently throughout, so split it up
        
        modified_flux = []
        for i in range(len(unique_vals)):
            unique_val = unique_vals[i]
            count = counts[i]
            index_start = ind[i]
            if count>1:
                indices = (sp == unique_val)
                wave = wvl[indices]
                fl = flux[indices]
                # Add in index where there was a jump between dw steps
                if np.any(index_start - ind == 1):
                    additional_ind = ind[np.where((index_start - ind == 1))[0][0]]
                    wave = np.append(wvl[additional_ind], wave)
                    fl = np.append(flux[additional_ind], fl)
                flux_ = rotBroad(wave, fl, epsilon, vsini)
                modified_flux.extend(flux_)
        return np.asarray(modified_flux)
    

def fastRotBroad(wvl, flux, epsilon, vsini, effWvl=None):
    """
    Apply rotational broadening using a single broadening kernel.

    The effect of rotational broadening on the spectrum is
    wavelength dependent, because the Doppler shift depends
    on wavelength. This function neglects this dependence, which
    is weak if the wavelength range is not too large.

    .. note:: numpy.convolve is used to carry out the convolution
              and "mode = same" is used. Therefore, the output
              will be of the same size as the input, but it
              will show edge effects.

    Parameters
    ----------
    wvl : array
        The wavelength
    flux : array
        The flux
    epsilon : float
        Linear limb-darkening coefficient
    vsini : float
        Projected rotational velocity in km/s.
    effWvl : float, optional
        The wavelength at which the broadening
        kernel is evaluated. If not specified,
        the mean wavelength of the input will be
        used.

    Returns
    -------
    Broadened spectrum : array
        The rotationally broadened output spectrum.
    """
    if vsini <= 0.0:
        raise ValueError("vsini must be positive.")
    if (epsilon < 0) or (epsilon > 1.0):
        raise ValueError("Linear limb-darkening coefficient, epsilon, should be '0 < epsilon < 1'.")
    
    # Check whether wavelength array is evenly spaced
    sp = wvl[1::] - wvl[0:-1]
    sp = np.append(abs(wvl[0] - wvl[1]), sp)
    sp = sp.round(decimals=5)
    
    unique_vals, ind, counts = np.unique(sp, return_index=True, return_counts=True)
    
    if len(unique_vals)==1:  # Wavelength array is evenly spaced
  
        # Wavelength binsize
        dwl = wvl[1] - wvl[0]

        if effWvl is None:
            effWvl = np.mean(wvl)

        gdl = _Gdl(vsini, epsilon)

        # The number of bins needed to create the broadening kernel
        binnHalf = int(np.floor(((vsini / 299792.458) * effWvl / dwl))) + 1
        gwvl = (np.arange(4*binnHalf) - 2*binnHalf) * dwl + effWvl
        # Create the broadening kernel
        dl = gwvl - effWvl
        g = gdl.gdl(dl, effWvl, dwl)
        # Remove the zero entries
        indi = np.where(g > 0.0)[0]
        g = g[indi]

        result = np.convolve(flux, g, mode="same") * dwl
        return result
    
    elif len(unique_vals) > 1:  # Wavelength array is sampled differently throughout

        modified_flux = []
        for i in range(len(unique_vals)):
            unique_val = unique_vals[i]
            count = counts[i]
            index_start = ind[i]
            if count>1:
                indices = (sp == unique_val)
                wave = wvl[indices]
                fl = flux[indices]
                # Add in index where there was a jump between dw steps
                if np.any(index_start - ind == 1):
                    additional_ind = ind[np.where((index_start - ind == 1))[0][0]]
                    wave = np.append(wvl[additional_ind], wave)
                    fl = np.append(flux[additional_ind], fl)
                flux_ = fastRotBroad(wave, fl, epsilon, vsini)
                modified_flux.extend(flux_)
        return np.asarray(modified_flux)


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


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """ 
     
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
        

    if window_len<3:
        return x
    
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
