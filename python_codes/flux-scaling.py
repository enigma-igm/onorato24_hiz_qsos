# Code to loop over the entire sample, scale all of them to the available magnitude and calculate the M1450 from the spectrum

import os
import numpy as np
from matplotlib import pyplot as plt, gridspec
import matplotlib.ticker as ticker
from astropy.io import ascii, fits
from astropy.cosmology import Planck18 as cosmo
from IPython import embed
from highz_qso_redux.utils import utils
from highz_qso_redux.utils.utils import open_spectra
from pypeit.utils import inverse, fast_running_median
from highz_qso_redux.utils.photo_utils import add_filter, calculate_flux_from_AB_magnitude, calculate_flux_error_from_AB_magnitude_error, scale_mag, from_vega_to_ab
from highz_qso_redux.scripts.flux_cal.Jinyi_flux_cal_functions import m1450
from qso_fitting.data.fluxing.flux_correct import spec_interp_gpm, filter_wave_ranges
from qso_fitting.analysis.magnitude_conversions import mag_to_Mlam_single

# Define the path to the spectra
spectra_dir = "/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/PypeIt_data/REDUX_OUT/spectra_for_plotting_coadd"

# Define the path to the quasar list saved in a csv file
qso_list = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/magnitudes/J_mag_list.csv'
filters_path = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/filters/'
# Read the csv file
qso_table = ascii.read(qso_list, format='basic', delimiter=',', names=['qso_name','z','err_z','mag_J','err_J','system','passband','keyword','M1450_tab','conv_fact','mag_Y','err_Y','mag_J_tab(Vega)','mag_Kp','err_Kp'], guess=False, fast_reader=True)

# Sort the table by decreasing redshift
qso_table.sort("z", reverse=True)

# Create a dictionary to store the spectra and their redshifts for each quasar
spectra_dict = {}

for filename in os.listdir(spectra_dir):
    if filename.endswith(".fits"):
        qso_name1 = filename[1:5]
        qso_name2 = filename[6:10]
        # Look for the quasar in the table
        for row in qso_table:
            if qso_name1 in row["qso_name"] and qso_name2 in row["qso_name"]:
                if row["qso_name"] not in spectra_dict:
                    spectra_dict[row["qso_name"]] = []
                spectra_dict[row["qso_name"]].append((filename, row["z"], row["err_z"], row["mag_J"], row["err_J"], row["system"], row["passband"], row["keyword"], row["M1450_tab"], row["conv_fact"], row["mag_Y"], row["err_Y"], row["mag_J_tab(Vega)"], row["mag_Kp"], row["err_Kp"]))
                break

# Sort the dictionary by decreasing redshift
spectra_dict = dict(sorted(spectra_dict.items(), key=lambda x: x[1][0][1], reverse=True))

# -------------------------------------------------------------------------------
# Define the terms to exclude from the plot
excluded_keys = []
excluded_values = []

spectra_dict_new = {}
for key in spectra_dict.keys():
    if any(term in spectra_dict[key][0][0] for term in excluded_keys):
        pass
    else:
        spectra_dict_new[key] = spectra_dict[key]

for key in spectra_dict_new.keys():
    if len(spectra_dict_new[key]) > 1 and spectra_dict_new[key][0][0][0:10] == spectra_dict_new[key][1][0][0:10]:
        for term in excluded_values:
            if term in spectra_dict_new[key][0][0]:
                x = [spectra_dict_new[key][1]]
                spectra_dict_new[key] = x
                break
            elif term in spectra_dict_new[key][1][0]:
                x = [spectra_dict_new[key][0]]
                spectra_dict_new[key] = x
                break

spectra_all = list(spectra_dict_new.values())

debug = True
save_spec = True
save_table = True

M_1450_spec_list = []
M_1450_phot_list = []
J_ab_list = []

# The following M1450 values are taken from the bal_continuum.py code because the M1450 from the spectrum is not reliable (BAL features)
M_0910 = -26.396787
M_0923 = -26.523945

# Create a list containing only the short names of the quasars: from qso[0:10]
qso_short_list = []
# -------------------------------------------------------------------------------
# Loop over the spectra and scale them to the J band magnitude
for i, spectra_list in enumerate(spectra_all):
    qso, z, err_z, mag_J, err_J, system, passband, keyword, M1450_tab, conv_fact, mag_Y, err_Y, mag_J_tab, mag_Kp, err_Kp = spectra_list[0]
    wave, flux, ivar, gpm, telluric = open_spectra(spectra_dir, qso, telluric=True)
    flux_interp, ivar_interp = spec_interp_gpm(wave, flux, gpm, sigma_or_ivar=ivar)
    # Add the short name of the quasar to the list
    qso_short_list.append(qso[0:10])

    # Photometric info for the quasar
    if system == 'Vega':
        J_ab = from_vega_to_ab(mag_J, conv_fact)
        J_ab_list.append(J_ab)
    else:
        J_ab = mag_J
        J_ab_list.append(J_ab)

    # Handle problematic cases
    if np.ma.is_masked(mag_J) and np.ma.is_masked(mag_Y) and np.ma.is_masked(mag_Kp):
        m_1450_spec, M_1450_spec = m1450(wave, flux, z, cosmo)
        M_1450_spec = float(M_1450_spec)

        M_1450_spec_list.append(M_1450_spec)
        M_1450_phot_list.append(0.0)
        if debug == True:
            print('No photometry available for %s' %qso[0:10] + ': M1450 from non-scaled spectrum!')
        continue

    elif np.ma.is_masked(mag_J) and not np.ma.is_masked(mag_Y):
        if passband == 'UKIRT' and keyword == 'WFCAM':
            filter_Y, eff_wave_Y = add_filter(filters_path+'UKIRT_WFCAM.Y.dat', 'UKIRT', 'Y')
            Y_passband = np.genfromtxt(filters_path+'UKIRT_WFCAM.Y.dat', dtype=None, encoding=None)

        elif passband == 'Subaru' and keyword == 'HSC':
            filter_Y, eff_wave_Y = add_filter(filters_path+'Subaru_HSC.Y.dat', 'Subaru', 'Y')
            Y_passband = np.genfromtxt(filters_path+'Subaru_HSC.Y.dat', dtype=None, encoding=None)

        wave_Y_16, wave_Y_50, wave_Y_84 = filter_wave_ranges(filter_Y)

        flux_scaled, ivar_scaled, scale_factor = scale_mag(wave, flux_interp, ivar_interp, filter_Y, mag_Y)
        Y_flux_speclite = calculate_flux_from_AB_magnitude(mag_Y, eff_wave_Y)
        err_Y_flux_lambda = calculate_flux_error_from_AB_magnitude_error(mag_Y, err_Y, eff_wave_Y)

        # Calculate the M1450 from the spectrum
        m_1450_spec, M_1450_spec = m1450(wave, flux_scaled, z, cosmo)
        # Be sure M1450 is a float
        M_1450_spec = float(M_1450_spec)
        M_1450_phot = 0.0

        M_1450_spec_list.append(M_1450_spec)
        M_1450_phot_list.append(M_1450_phot)

        if debug == True:
            print('The scale factor for %s in the Y-band is %f' % (qso[0:10], scale_factor))
            print('M1450 from the spectrum of %s is %f' % (qso[0:10], M_1450_spec))

    elif np.ma.is_masked(mag_J) and np.ma.is_masked(mag_Y) and not np.ma.is_masked(mag_Kp):
        if passband == 'Keck' and keyword == 'NIRC2':
            filter_Kp, eff_wave_Kp = add_filter(filters_path+'Keck_NIRC2.Kp.dat', 'Keck', 'Kp')
            Kp_passband = np.genfromtxt(filters_path+'Keck_NIRC2.Kp.dat', dtype=None, encoding=None)

        wave_Kp_16, wave_Kp_50, wave_Kp_84 = filter_wave_ranges(filter_Kp)

        flux_scaled, ivar_scaled, scale_factor = scale_mag(wave, flux_interp, ivar_interp, filter_Kp, mag_Kp)
        Kp_flux_speclite = calculate_flux_from_AB_magnitude(mag_Kp, eff_wave_Kp)
        err_Kp_flux_lambda = calculate_flux_error_from_AB_magnitude_error(mag_Kp, err_Kp, eff_wave_Kp)

        # Calculate the M1450 from the spectrum
        m_1450_spec, M_1450_spec = m1450(wave, flux_scaled, z, cosmo)
        # Be sure M1450 is a float
        M_1450_spec = float(M_1450_spec)
        M_1450_phot = 0.0

        M_1450_spec_list.append(M_1450_spec)
        M_1450_phot_list.append(M_1450_phot)

        if debug == True:
            print('The scale factor for %s in the Kp-band is %f' % (qso[0:10], scale_factor))
            print('M1450 from the spectrum of %s is %f' % (qso[0:10], M_1450_spec))

    elif not np.ma.is_masked(mag_J):
        # Load the transmission curve for the J band
        if passband == 'UKIRT' and keyword == 'WFCAM':
            filter_J, eff_wave_J = add_filter(filters_path+'UKIRT_WFCAM.J.dat', 'UKIRT', 'J')
            J_passband = np.genfromtxt(filters_path+'UKIRT_WFCAM.J.dat', dtype=None, encoding=None)
        elif passband == 'UKIRT' and keyword == 'UHS':
            filter_J, eff_wave_J = add_filter(filters_path+'UKIRT_UKIDSS.J.dat', 'UKIRT', 'J')
            J_passband = np.genfromtxt(filters_path+'UKIRT_UKIDSS.J.dat', dtype=None, encoding=None)
        elif passband == 'UKIRT' and keyword == None:
            filter_J, eff_wave_J = add_filter(filters_path+'UKIRT_UKIDSS.J.dat', 'UKIRT', 'J')
            J_passband = np.genfromtxt(filters_path+'UKIRT_UKIDSS.J.dat', dtype=None, encoding=None)
        elif passband == 'VISTA':
            filter_J, eff_wave_J = add_filter(filters_path+'Paranal_VISTA.J.dat', 'VISTA', 'J')
            J_passband = np.genfromtxt(filters_path+'Paranal_VISTA.J.dat', dtype=None, encoding=None)
        elif passband == 'NOTcam':
            filter_J, eff_wave_J = add_filter(filters_path+'NOT_NOTcam.J.dat', 'NOTcam', 'J')
            J_passband = np.genfromtxt(filters_path+'NOT_NOTcam.J.dat', dtype=None, encoding=None)
        elif passband == 'NTT' and keyword == 'SofI':
            filter_J, eff_wave_J = add_filter(filters_path+'LaSilla_SOFI.Js.dat', 'SofI', 'J')
            J_passband = np.genfromtxt(filters_path+'LaSilla_SOFI.Js.dat', dtype=None, encoding=None)

        wave_J_16, wave_J_50, wave_J_84 = filter_wave_ranges(filter_J)

        # Scale the spectrum to the J band magnitude
        flux_scaled, ivar_scaled, scale_factor = scale_mag(wave, flux_interp, ivar_interp, filter_J, J_ab)
        # Convert the magnitude to flux for plotting
        J_flux_speclite = calculate_flux_from_AB_magnitude(J_ab, eff_wave_J)
        err_J_flux_lambda = calculate_flux_error_from_AB_magnitude_error(J_ab, err_J, eff_wave_J)

        # Calculate the M1450 from the spectrum
        m_1450_spec, M_1450_spec = m1450(wave, flux_scaled, z, cosmo)
        # Calculate the M1450 from the photometry
        M_1450_phot = mag_to_Mlam_single(J_ab, 'J', z, 1450.0)

        # Be sure M1450 is a float
        M_1450_spec = float(M_1450_spec)
        M_1450_phot = float(M_1450_phot)

        if 'J0910-0414' in qso:
            print('M1450 from the spectrum of %s is %f and cannot be trusted because of BAL features. More reliable estimate is: %f' % (qso[0:10], M_1450_spec, M_0910))
            M_1450_spec = M_0910
        elif 'J0923+0402' in qso:
            print('M1450 from the spectrum of %s is %f and cannot be trusted because of BAL features. More reliable estimate is: %f' % (qso[0:10], M_1450_spec, M_0923))
            M_1450_spec = M_0923

        M_1450_spec_list.append(M_1450_spec)
        M_1450_phot_list.append(M_1450_phot)

        if debug == True:
            print('The scale factor for %s in the J-band is %f' % (qso[0:10], scale_factor))
            print('M1450 from the spectrum of %s is %f, while m1450 is %f' % (qso[0:10], M_1450_spec, m_1450_spec))
            print('M1450 from the J-band photometry of %s is %f' % (qso[0:10], M_1450_phot))

    if debug == True and flux_scaled is not None:
        # Plot the old and the new spectrum
        fx = plt.figure(figsize=(15, 7))
        gs = gridspec.GridSpec(12, 5, figure=fx, wspace=0., hspace=0.)
        ax = fx.add_subplot(gs[0:9, 0:5])
        ax2 = fx.add_subplot(gs[9:12, 0:5], sharex=ax)

        # Smooth for visualization
        window = 10
        flux_sm, ivar_sm = utils.ivarsmooth(flux_interp, ivar_interp, window=window)
        gpm_sm = (ivar_sm > 0.0)
        sigma_sm = np.sqrt(inverse(ivar_sm))

        new_flux_sm, new_ivar_sm = utils.ivarsmooth(flux_scaled, ivar_scaled, window=window)
        new_gpm_sm = (new_ivar_sm > 0.0)
        new_sigma_sm = np.sqrt(inverse(new_ivar_sm))

        ax.plot(wave, flux_sm, drawstyle='steps-mid', color='royalblue', zorder=9, label='Original spectrum', alpha=0.8)
        ax.plot(wave, new_flux_sm, drawstyle='steps-mid', color='darkorange', zorder=10, label='Scaled spectrum', alpha=0.8)
        ax.plot(wave, sigma_sm, drawstyle='steps-mid', color='midnightblue', zorder=5, alpha=0.6)
        ax.plot(wave, new_sigma_sm, drawstyle='steps-mid', color='peru', zorder=6, alpha=0.6)

        ax.text(0.8, 0.62, qso[0:10], transform=ax.transAxes, fontsize=18)

        # Plot the photometric point
        if np.ma.is_masked(mag_J) and not np.ma.is_masked(mag_Y):
            ax.errorbar(eff_wave_Y, Y_flux_speclite * 1e+17, yerr=err_Y_flux_lambda * 1e+17, fmt='o', zorder=15, label='Y-band mag', markersize=8, color='purple')

            if np.ma.is_masked(keyword):
                ax2.fill_between(Y_passband[:, 0], Y_passband[:, 1], color='purple', alpha=0.4, label=passband)
            else:
                ax2.fill_between(Y_passband[:, 0], Y_passband[:, 1], color='purple', alpha=0.4, label=passband+'/'+keyword)

        elif np.ma.is_masked(mag_J) and np.ma.is_masked(mag_Y) and not np.ma.is_masked(mag_Kp):
            ax.errorbar(eff_wave_Kp, Kp_flux_speclite * 1e+17, yerr=err_Kp_flux_lambda * 1e+17, fmt='o', zorder=15, label=r'$\rm{K_{p}}$-band mag', markersize=8, color='darkred')

            if np.ma.is_masked(keyword):
                ax2.fill_between(Kp_passband[:, 0], Kp_passband[:, 1], color='darkred', alpha=0.4, label=passband)
            else:
                ax2.fill_between(Kp_passband[:, 0], Kp_passband[:, 1], color='darkred', alpha=0.4, label=passband+'/'+keyword)

        elif not np.ma.is_masked(mag_J):
            ax.errorbar(eff_wave_J, J_flux_speclite * 1e+17, yerr=err_J_flux_lambda * 1e+17, fmt='o', zorder=15, label='J-band mag', markersize=8, color='navy')

            if np.ma.is_masked(keyword):
                ax2.fill_between(J_passband[:, 0], J_passband[:, 1], color='navy', alpha=0.4, label=passband)
            else:
                ax2.fill_between(J_passband[:, 0], J_passband[:, 1], color='navy', alpha=0.4, label=passband+'/'+keyword)

        ax2.set_xlabel(r'Observed Wavelength [$\rm{\AA}$]', size=20)
        ax.set_ylabel(r'Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]', size=19)
        ax2.set_ylabel(r'Transmission', size=18)
        x_min = 8830.
        x_max = 24990.
        if 'XShooter' in qso:
            ymax = (fast_running_median(flux_sm, 30)).max()
            ymax_new = (fast_running_median(new_flux_sm, 30)).max()
            ymax_xsh = 1.3 * min(ymax, flux_sm[(wave > 1215.67 * (1 + z) - 20.0) & (wave < 1215.67 * (1 + z) + 20.0)].max())
            ymax_xsh_new = 1.3 * min(ymax_new, new_flux_sm[(wave > 1215.67 * (1 + z) - 20.0) & (wave < 1215.67 * (1 + z) + 20.0)].max())
            if ymax_xsh_new > ymax_xsh:
                ymax_xsh = ymax_xsh_new
            ax.axis([x_min, x_max, -0.02, ymax_xsh])
        else:
            ymax1 = 1.1 * np.max(flux_sm * gpm_sm)
            ymax1_new = 1.1 * np.max(new_flux_sm * new_gpm_sm)
            if ymax1_new > ymax1:
                ymax1 = ymax1_new
            ax.axis([x_min, x_max, -0.02, ymax1])

        ax.tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=18)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
        ax2.tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=18)
        ax.legend(fontsize=18, loc='upper right', frameon=True)
        if np.ma.is_masked(mag_J) and np.ma.is_masked(mag_Y) and not np.ma.is_masked(mag_Kp):
            ax2.axis([x_min, x_max, -0.02, 0.99])
            ax2.legend(fontsize=18, loc='upper left', frameon=True)
        else:
            ax2.legend(fontsize=18, loc='upper right', frameon=True)
        plt.tight_layout()
        # plt.savefig('../../figures/scaled_spectra/scale_spec_%s.png' % qso[0:10], dpi=300)
        plt.show()
        plt.close()
# -------------------------------------------------------------------------------
# Save the new flux_scaled and ivar_scaled in a new fits file with these 2 more columns
    if save_spec == True and flux_scaled is not None:
        hdu=fits.open(spectra_dir+"/"+qso, fix = True, ignore_missing_simple=True)
        spec=hdu[1].data
        wave, wave_grid_mid, flux, ivar, gpm, telluric = spec['wave'], spec['wave_grid_mid'], spec['flux'], spec['ivar'], np.array(spec['mask'], dtype=bool), spec['telluric']
        if 'sigma' in spec.dtype.names:
            sigma = spec['sigma']
        elif 'obj_model' in spec.dtype.names:
            obj_model = spec['obj_model']

        #Add the new columns and save the new fits file depending on the photometry available
        if np.ma.is_masked(mag_J) and not np.ma.is_masked(mag_Y):
            new_cols = fits.ColDefs([fits.Column(name='flux_scaled_Y', format='D', array=flux_scaled),
                                     fits.Column(name='ivar_scaled_Y', format='D', array=ivar_scaled)])
            new_hdu = fits.BinTableHDU.from_columns(spec.columns + new_cols)

            new_hdu.writeto(spectra_dir+"/new_spec/"+qso, overwrite=True)

        elif np.ma.is_masked(mag_J) and np.ma.is_masked(mag_Y) and not np.ma.is_masked(mag_Kp):
            new_cols = fits.ColDefs([fits.Column(name='flux_scaled_Kp', format='D', array=flux_scaled),
                                     fits.Column(name='ivar_scaled_Kp', format='D', array=ivar_scaled)])
            new_hdu = fits.BinTableHDU.from_columns(spec.columns + new_cols)

            new_hdu.writeto(spectra_dir+"/new_spec/"+qso, overwrite=True)

        elif not np.ma.is_masked(mag_J):
            new_cols = fits.ColDefs([fits.Column(name='flux_scaled_J', format='D', array=flux_scaled),
                                     fits.Column(name='ivar_scaled_J', format='D', array=ivar_scaled)])
            new_hdu = fits.BinTableHDU.from_columns(spec.columns + new_cols)

            new_hdu.writeto(spectra_dir+"/flux_scaled/"+qso, overwrite=True)

# -------------------------------------------------------------------------------
# Create a new cvs file containing only: the name of the quasars, z, the J_ab mag, the error on the mag, M_1450_spec and M_1450_phot
if save_table == True and flux_scaled is not None:
    new_table = qso_table.copy()
    new_table.remove_columns(['mag_J','system', 'conv_fact', 'mag_J_tab(Vega)'])
    new_table.add_column(J_ab_list, name='J_ab', index=3)
    new_table.add_column(M_1450_spec_list, name='M_1450_spec', index=12)
    new_table.add_column(M_1450_phot_list, name='M_1450_phot', index=13)
    new_table.add_column(qso_short_list, name='short_name', index=14)

    # Save the new table
    new_table.write('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/magnitudes/mags_M1450.csv', format='csv', overwrite=True)