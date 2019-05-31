from astropy.io import fits as pyfits
import numpy as np


def get_synth_wavegrid(grid_name, synth_wave_file):
    """
    This function will grab the wavelength grid of the synthetic spectral grid you're working with.

    :param grid_name: either phoenix, intrigoss, or ambre
    :param synth_wave_file: the name of the file which contains the wavelength grid

    :return: Wavelength grid
    """

    if grid_name == 'intrigoss':
        # For INTRIGOSS spectra, the wavelength array is stored in the same file as the spectra
        hdulist = pyfits.open(synth_wave_file)
        wave_grid_synth = hdulist[1].data['wavelength']
    elif grid_name == 'phoenix':
        # For Phoenix spectra, the wavelength array is stored in a separate file
        hdulist = pyfits.open(synth_wave_file)
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
        wave_grid_synth = np.genfromtxt(synth_wave_file, usecols=0)
    else:
        raise ValueError('{} not a valid grid name. Need to supply an appropriate spectral grid name '
                         '(phoenix, intrigoss, or ambre)'.format(grid_name))

    return wave_grid_synth


def get_synth_spec_data(data_filename, grid_name='phoenix'):
    """
    Given the names of a file  and the grid of synthetic spectra you're working with (phoenix, intrigoss, ambre),
    this function will grab the flux and stellar parameters

    :param data_filename: Name of spectrum file
    :param grid_name: Name of spectral grid (phoenix, intrigoss, ambre)

    :return: flux and stellar parameters of spectrum file
    """

    if grid_name == 'phoenix':
        with pyfits.open(data_filename) as hdulist:
            flux = hdulist[0].data
            param_data = hdulist[0].header
            teff = param_data['PHXTEFF']
            logg = param_data['PHXLOGG']
            m_h = param_data['PHXM_H']
            a_m = param_data['PHXALPHA']
            vt = param_data['PHXXI_L']
            params = [teff, logg, m_h, a_m, vt]
    elif grid_name == 'intrigoss':
        with pyfits.open(data_filename) as hdulist:
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
        flux = np.genfromtxt(data_filename, usecols=2)

        # s = os.path.basename(filename)
        # flux_ = data[:,1]
        # s = os.path.basename(filename)

    return flux, params


def ensure_constant_sampling(wav):
    """
    This function will check if the wavelength array is evenly sampled, and modify it to be evenly sampled
    if not already.

    :param wav: Wavelength array
    :return: Evenly sampled wavelength array
    """

    sp = wav[1::] - wav[0:-1]
    sp = np.append(abs(wav[0] - wav[1]), sp)
    sp = sp.round(decimals=5)

    unique_vals, ind, counts = np.unique(sp, return_index=True, return_counts=True)

    if len(unique_vals) > 1:  # Wavelength array is sampled differently throughout

        # Rebin fluxes to the coarsest sampling found in the wavelength grid
        coarsest_sampling = max(unique_vals[counts > 1])
        new_wave_grid = np.arange(wav[0], wav[-1], coarsest_sampling)

        wav = new_wave_grid
    else:
        print('Wavelength array already has constant sampling! Returning same wavelength grid')

    return wav


if __name__ == '__main__':
    pass
