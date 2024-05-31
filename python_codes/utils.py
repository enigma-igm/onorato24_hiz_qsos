# Description: Utility functions
from astropy.io import fits
from qso_fitting.data.fluxing.flux_correct import spec_interp_gpm
import math
import numpy as np
import speclite.filters
from speclite.filters import ab_reference_flux
import astropy.units as u
from astropy.io import ascii
from astropy import constants as const
from IPython import embed
from jwst_lightecho.phot_selection.phot_utils import ABmag_to_nJy, nJy_to_ABmag

# Define some global constants that will be used in this module
c_light = (const.c.to('km/s')).value

def ivarsmooth(flux, ivar, window):
    """
    Boxcar smoothing of width window with ivar weights

    Args:
        flux: flux
        ivar: inverse variance
        window: window

    Returns:
        smoothflux: smoothed flux
        outivar: output inverse variance
    """

    nflux = (flux.shape)[0]
    halfwindow = int(np.floor((np.round(window) - 1)/2))
    shiftarr = np.zeros((nflux, 2*halfwindow + 1))
    shiftivar = np.zeros((nflux, 2*halfwindow + 1))
    shiftindex = np.zeros((nflux, 2*halfwindow + 1))
    indexarr = np.arange(nflux)
    indnorm = np.outer(indexarr,(np.zeros(2 *halfwindow + 1) + 1))

    for i in np.arange(-halfwindow,halfwindow + 1,dtype=int):
        shiftarr[:,i+halfwindow] = np.roll(flux,i)
        shiftivar[:, i+halfwindow] = np.roll(ivar, i)
        shiftindex[:, i+halfwindow] = np.roll(indexarr, i)

    wh = (np.abs(shiftindex - indnorm) > (halfwindow+1))
    shiftivar[wh]=0.0

    outivar = np.sum(shiftivar,axis=1)
    nzero, = np.where(outivar > 0.0)
    zeroct=len(nzero)
    smoothflux = np.sum(shiftarr * shiftivar, axis=1)
    if(zeroct > 0):
        smoothflux[nzero] = smoothflux[nzero]/outivar[nzero]
    else:
        smoothflux = np.roll(flux, 2*halfwindow + 1) # kill off NAN's

    return smoothflux, outivar

# -------------------------------------------------------------------------------
def open_spectra(path, spectrum, telluric=None, flux_scale=None):
    """
    Open the spectra

    Args:
        path: path to the spectra
        spectrum: spectrum to open

    Returns:
        wave: wavelength
        flux: flux
        ivar: inverse variance
        gpm: good pixel mask
        telluric: telluric transmission

    """
    hdu=fits.open(path+"/"+spectrum, fix = True, ignore_missing_simple=True)
    spec=hdu[1].data
    # Standard spectra ground based
    if 'wave_grid_mid' in spec.dtype.names and telluric is not None and flux_scale is None:
        wave, flux, ivar, gpm, telluric = spec['wave_grid_mid'], spec['flux'], spec['ivar'], np.array(spec['mask'], dtype=bool), spec['telluric']
    # Standard spectra from space
    elif 'wave_grid_mid' in spec.dtype.names and 'F_lam' in spec.dtype.names and telluric is None and flux_scale is None:
        wave, flux, ivar, gpm = spec['wave_grid_mid'], spec['F_lam'], spec['ivar'], np.array(spec['mask'], dtype=bool)
    # Spectra not telluric corrected yet
    elif 'wave_grid_mid' in spec.dtype.names and 'F_lam' not in spec.dtype.names and telluric is None and flux_scale is None:
        wave, flux, ivar, gpm = spec['wave_grid_mid'], spec['flux'], spec['ivar'], np.array(spec['mask'], dtype=bool)
    # Old spectra from previous versions of PypeIt
    elif 'OPT_WAVE' in spec.dtype.names and telluric is not None and flux_scale is None:
        wave, flux, ivar, gpm, telluric = spec['OPT_WAVE'].data, spec['OPT_FLAM'].data, spec['OPT_FLAM_IVAR'].data, np.array(spec['OPT_MASK'].data, dtype=bool), spec['telluric'].data
        flux = np.array(flux, dtype=np.float64)
        ivar = np.array(ivar, dtype=np.float64)
        telluric = np.array(telluric, dtype=np.float64)
    elif 'OPT_WAVE' in spec.dtype.names and telluric is None and flux_scale is None:
        wave, flux, ivar, gpm = spec['OPT_WAVE'].data, spec['OPT_FLAM'].data, spec['OPT_FLAM_IVAR'].data, np.array(spec['OPT_MASK'].data, dtype=bool)
        flux = np.array(flux, dtype=np.float64)
        ivar = np.array(ivar, dtype=np.float64)
    # Flux from fit with more mags
    elif 'flux_fit' in spec.dtype.names and telluric is not None and flux_scale is not None:
        wave, flux_fit, ivar_fit, gpm, telluric = spec['wave_grid_mid'].data, spec['flux_fit'].data, spec['ivar_fit'].data, np.array(spec['mask'].data, dtype=bool), spec['telluric'].data
        flux_fit = np.array(flux_fit, dtype=np.float64)
        ivar_fit = np.array(ivar_fit, dtype=np.float64)
    # Flux scaled spectra ground based
    elif 'flux_scaled_J' in spec.dtype.names and telluric is not None and flux_scale is not None:
        wave, flux_scaled, ivar_scaled, gpm, telluric = spec['wave_grid_mid'].data, spec['flux_scaled_J'].data, spec['ivar_scaled_J'].data, np.array(spec['mask'].data, dtype=bool), spec['telluric'].data
        flux_scaled = np.array(flux_scaled, dtype=np.float64)
        ivar_scaled = np.array(ivar_scaled, dtype=np.float64)
    elif 'flux_scaled_Y' in spec.dtype.names and telluric is not None and flux_scale is not None:
        wave, flux_scaled, ivar_scaled, gpm, telluric = spec['wave_grid_mid'].data, spec['flux_scaled_Y'].data, spec['ivar_scaled_Y'].data, np.array(spec['mask'].data, dtype=bool), spec['telluric'].data
        flux_scaled = np.array(flux_scaled, dtype=np.float64)
        ivar_scaled = np.array(ivar_scaled, dtype=np.float64)
    elif 'flux_scaled_Kp' in spec.dtype.names and telluric is not None and flux_scale is not None:
        wave, flux_scaled, ivar_scaled, gpm, telluric = spec['wave_grid_mid'].data, spec['flux_scaled_Kp'].data, spec['ivar_scaled_Kp'].data, np.array(spec['mask'].data, dtype=bool), spec['telluric'].data
        flux_scaled = np.array(flux_scaled, dtype=np.float64)
        ivar_scaled = np.array(ivar_scaled, dtype=np.float64)
    # If I loop over mixed spectra (some flux scaled and some not) I need to add the following two elifs
    elif 'flux_scaled_J' not in spec.dtype.names and telluric is not None and flux_scale is not None:
        wave, flux, ivar, gpm, telluric = spec['wave_grid_mid'].data, spec['flux'].data, spec['ivar'].data, np.array(spec['mask'].data, dtype=bool), spec['telluric'].data
        flux = np.array(flux, dtype=np.float64)
        ivar = np.array(ivar, dtype=np.float64)
    elif 'flux_scaled_Y' not in spec.dtype.names and telluric is not None and flux_scale is not None:
        wave, flux, ivar, gpm, telluric = spec['wave_grid_mid'].data, spec['flux'].data, spec['ivar'].data, np.array(spec['mask'].data, dtype=bool), spec['telluric'].data
        flux = np.array(flux, dtype=np.float64)
        ivar = np.array(ivar, dtype=np.float64)
    elif 'flux_scaled_Kp' not in spec.dtype.names and telluric is not None and flux_scale is not None:
        wave, flux, ivar, gpm, telluric = spec['wave_grid_mid'].data, spec['flux'].data, spec['ivar'].data, np.array(spec['mask'].data, dtype=bool), spec['telluric'].data
        flux = np.array(flux, dtype=np.float64)
        ivar = np.array(ivar, dtype=np.float64)
    else:
        wave, flux, ivar, gpm = spec['wave'].data, spec['flux'].data, spec['ivar'].data, np.array(spec['mask'].data, dtype=bool)
    wave = np.array(wave, dtype=np.float64)

    if telluric is not None and flux_scale is None:
        return wave, flux, ivar, gpm, telluric
    elif telluric is None and flux_scale is None:
        return wave, flux, ivar, gpm
    elif 'flux_fit' in spec.dtype.names and telluric is not None and flux_scale is not None:
        return wave, flux_fit, ivar_fit, gpm, telluric
    elif 'flux_scaled_J' in spec.dtype.names and telluric is not None and flux_scale is not None:
        return wave, flux_scaled, ivar_scaled, gpm, telluric
    elif 'flux_scaled_Y' in spec.dtype.names and telluric is not None and flux_scale is not None:
        return wave, flux_scaled, ivar_scaled, gpm, telluric
    elif 'flux_scaled_Kp' in spec.dtype.names and telluric is not None and flux_scale is not None:
        return wave, flux_scaled, ivar_scaled, gpm, telluric
    elif 'flux_scaled_J' not in spec.dtype.names and telluric is not None and flux_scale is not None:
        return wave, flux, ivar, gpm, telluric
    elif 'flux_scaled_Y' not in spec.dtype.names and telluric is not None and flux_scale is not None:
        return wave, flux, ivar, gpm, telluric
    elif 'flux_scaled_Kp' not in spec.dtype.names and telluric is not None and flux_scale is not None:
        return wave, flux, ivar, gpm, telluric
    else:
        return wave, flux, ivar, gpm

# -------------------------------------------------------------------------------
def wave_0(wave, z):
    """
    Move wavelength to the rest-frame

    Args:
        wave: wavelength
        z: redshift

    Returns:
        wave_lambda0: wavelength in the rest-frame

    """
    wave_lambda0 = wave/(1.0+z)
    return wave_lambda0

# -------------------------------------------------------------------------------
def flux_0(flux, ivar, z):
    """
    Move flux and ivar to the rest-frame.

    Args:
        flux: flux
        ivar: inverse variance
        z: redshift

    Returns:
        flux_lambda0: flux in the rest-frame
        ivar_lambda0: ivar in the rest-frame

    """
    flux_lambda0 = flux*(1.0+z)
    ivar_lambda0 = ivar/(1.0+z)**2
    return flux_lambda0, ivar_lambda0

# -------------------------------------------------------------------------------
def normalize_spectra(wave, flux, ivar, wave_norm):
    """
    Normalize the spectra to be unity at wavelength of normalization.

    Args:
        wave: wavelength
        flux: flux
        ivar: inverse variance
        wave_norm: wavelength to normalize to

    Returns:
        flux_norm: normalized flux

    """
    idx = np.argmin(np.abs(wave - wave_norm))
    flux_norm = flux / np.mean(flux[idx - 25:idx + 25])
    ivar_norm = ivar * np.mean(flux[idx - 25:idx + 25])**2

    return flux_norm, ivar_norm

# -------------------------------------------------------------------------------
def compute_snr(wave, flux, ivar, gpm, wave_range):
    """
    Compute the SNR in a given wavelength range

    Parameters
    ----------
    wave (np.ndarray):
        wavelength array
    flux (np.ndarray):
        flux array
    ivar (np.ndarray):
        inverse variance array
    wave_range (tuple):
        wavelength range to compute the SNR

    Returns
    -------
    result (tuple):
        mean and median SNR
    """
    flux_interp, ivar_interp = spec_interp_gpm(wave, flux, gpm, sigma_or_ivar=ivar)
    sn_ratio = flux_interp * np.sqrt(ivar_interp)
    wavemask = (wave > wave_range[0]) & (wave < wave_range[1]) & (ivar > 0.0)
    mean_snr, median_snr = np.mean(sn_ratio[wavemask]), np.median(sn_ratio[wavemask])
    result = (mean_snr, median_snr)

    return result

# -------------------------------------------------------------------------------
def wave_obs(wave_lambda0, z):
    """
    Move wavelength to the observed frame

    Args:
        wave_lambda0: wavelength in the rest-frame
        z: redshift

    Returns:
        wave: wavelength in the observed frame

    """
    wave = wave_lambda0*(1.0+z)
    return wave

# -------------------------------------------------------------------------------
def add_filter(filter_path, group_name, band_name):
    """
    Utility routine to add a filter to speclite.
    Args:
        filter_path: path to the filter file
        group_name: name of the group of filters
        band_name: band name

    Returns:
        filter: speclite filter
        eff_wave: effective wavelength of the filter
    """
    filter_name = ascii.read(filter_path)

    filter_name['col1'] = filter_name['col1'].astype(float)
    filter_name['col2'] = filter_name['col2'].astype(float)

    dwave_list = []
    for i in range(len(filter_name['col1'])):
        dwave = np.abs(filter_name['col1'][i] - filter_name['col1'][i - 1]) # Calculate the difference between two consecutive wavelengths
        dwave_list.append(dwave)
        if filter_name['col2'][i] < 0.0: # If the transmission is negative, set it to 0.0
            filter_name['col2'][i] = 0.0
    dwave_final = np.median(dwave_list)
    if filter_name['col2'][0] > 0.0:
        filter_name.insert_row(0, [filter_name['col1'][0] - dwave_final/1000., 0.0]) # Add 0.0 transmission at the beginning
    if filter_name['col2'][-1] > 0.0:
        filter_name.add_row([filter_name['col1'][-1] + dwave_final/1000., 0.0]) # Add 0.0 transmission at the end

    if group_name == 'UKIRT' or group_name == 'VISTA' or group_name == 'NOTcam' or group_name == 'Subaru' or group_name == 'PANSTARRS' or group_name == 'Keck':
        F_name = speclite.filters.FilterResponse(wavelength=filter_name['col1'] * u.AA, response=filter_name['col2'],
                                              meta=dict(group_name=group_name, band_name=band_name))
    if group_name == 'VISTA_ETC':
        filter_name['col1'] = filter_name['col1'] * 10. # Convert the wavelength from nm to AA
        filter_name['col2'] = filter_name['col2'] / 100. # Convert the transmission from % to fraction
        F_name = speclite.filters.FilterResponse(wavelength=filter_name['col1'] * u.AA, response=filter_name['col2'],
                                              meta=dict(group_name=group_name, band_name=band_name))
    if group_name == 'SofI':
        unique_wavelengths, unique_indices = np.unique(filter_name['col1'], return_index=True)
        filter_name = filter_name[unique_indices]
        interpolated_transmissivity = np.interp(filter_name['col1'], unique_wavelengths, filter_name['col2'])

        if interpolated_transmissivity[0] != 0:
            extra_lambda_min = unique_wavelengths[0] - 0.0001
            unique_wavelengths = np.insert(unique_wavelengths, 0, extra_lambda_min)
            interpolated_transmissivity = np.insert(interpolated_transmissivity, 0.0)

        elif interpolated_transmissivity[-1] != 0:
            extra_lambda_max = unique_wavelengths[-1] + 0.0001
            unique_wavelengths = np.append(unique_wavelengths, extra_lambda_max)
            interpolated_transmissivity = np.append(interpolated_transmissivity, 0.0)
        F_name = speclite.filters.FilterResponse(wavelength=unique_wavelengths * u.AA, response=interpolated_transmissivity,
                                                meta=dict(group_name=group_name, band_name=band_name))

    filter = speclite.filters.load_filters(group_name+'-'+band_name)

    eff_wave = filter.effective_wavelengths.value[0] # AA

    return filter, eff_wave

# -------------------------------------------------------------------------------
def calculate_AB_magnitude(flux_density, passband, wavelength):

    mag_AB = passband.get_ab_magnitudes(flux_density * 1e-17 * u.erg / u.s / u.cm**2 / u.AA, wavelength *u.AA)[0][0]
    maggies = passband.get_ab_maggies(flux_density * 1e-17 * u.erg / u.s / u.cm**2 / u.AA, wavelength *u.AA)

    return mag_AB, maggies

# -------------------------------------------------------------------------------
def calculate_flux_from_AB_magnitude(AB_magnitude, lambda_eff):
    # Calculate the flux from the AB magnitude
    # flux_nu = 10 ** (-0.4 * (AB_magnitude + 48.6))
    # Convert this F(nu) to F(lambda)
    # flux_lambda = flux_nu * const.c.to('m/s').value * 10 ** 10 / lambda_eff ** 2

    flux_speclite = ab_reference_flux(lambda_eff * u.AA, magnitude=AB_magnitude) # erg/s/cm^2/A

    return flux_speclite.value

# -------------------------------------------------------------------------------
def calculate_flux_error_from_AB_magnitude_error(AB_magnitude, err_AB_magnitude, lambda_eff):
    err_flux_nu = 0.4 * math.log(10) * 10**(-0.4 * (AB_magnitude + 48.6)) * err_AB_magnitude
    err_flux_lambda = err_flux_nu * const.c.to('m/s').value * 10 ** 10 / lambda_eff ** 2

    return err_flux_lambda

# -------------------------------------------------------------------------------
def scale_mag(wave, flux, ivar, passband, true_mag):
    """
    Scale the spectrum to the given magnitude in the given passband
    Args:
        wave: wavelength in units of A
        flux: flux in units of erg/s/cm2/A
        ivar: inverse variance of the flux
        passband: name of the passband
        true_mag: true magnitude in the given passband
    Returns:
        flux_scaled: flux in units of erg/s/cm2/A scaled to the given magnitude
        ivar_scaled: ivar scaled to the given magnitude
        scale_factor: scale factor to apply to the flux and the ivar
    """

    mag_spec, maggie_spec = calculate_AB_magnitude(flux, passband, wave)
    scale_factor = 10 ** (0.4 * (mag_spec - true_mag))

    # Scale the flux and the ivar
    flux_scaled = flux * scale_factor
    ivar_scaled = ivar / scale_factor**2

    return flux_scaled, ivar_scaled, scale_factor

# -------------------------------------------------------------------------------
def from_vega_to_ab(mag_vega, offset):
    """
    Convert Vega magnitudes to AB magnitudes.

    Parameters
    ----------
    mag : float
        Vega magnitude.
    offset : float
        Offset to convert Vega to AB magnitudes.

    Returns
    -------
    mag_ab : float
        AB magnitude.
    """
    mag_ab = mag_vega + offset

    return mag_ab

# -------------------------------------------------------------------------------
def ABmagerr_to_nJyerr(dm, m):
    """
    Convert AB magnitude error to nJy error

    Args:
    -----
    dm (float or np.ndarray):
       AB magnitude error
    m (float or np.ndarray):
       flux as an AB magnitude

    Returns:
    -------
    nJy_err (float or np.ndarray):
         nJy error
    """

    return ABmag_to_nJy(m)*np.log(10.0)/2.5*dm

# -------------------------------------------------------------------------------
def m1450(wave,flux,redshift,cosmo):
  """
  :param wave: observed wavelength
  :param flux: observed flux in units of erg/s/cm2/A
  :param redshift: object redshift
  :return: apparent and absolute AB magnitude at rest-frame 1450A.
  """
  wave_rest = wave_0(wave, redshift)
  idx1450 = np.where((wave_rest > 1445.0) & (wave_rest < 1455.0))
  flux1450 = np.median(flux[idx1450]) ## erg/s/cm2/A

  flux1450 = flux1450 * 1e-17 * u.erg / u.s / u.cm**2 / u.AA
  fnu1450 = flux1450.to(u.Jy, equivalencies=u.spectral_density(1450.0*(1+redshift)*u.AA))
  m_1450 = -2.5 * np.log10(fnu1450.value) + 8.9
  Lnu = (4.0 * np.pi * np.square(cosmo.luminosity_distance(redshift))/ (1.0 + redshift) * fnu1450).decompose().to('erg/s/Hz')
  M_1450 = -2.5 * np.log10((Lnu/(4.0*np.pi*np.square(10.0*u.pc))/(3631.0*u.Jy)).decompose())

  return m_1450, M_1450