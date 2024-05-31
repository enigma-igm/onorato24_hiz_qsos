# Script to compute the SNR in different wavelength ranges (J, H, K bands) for the spectra of the quasars
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
from collections import defaultdict
from pypeit.utils import inverse, fast_running_median
from qso_fitting.data.utils import get_wave_grid, rebin_spectra
from highz_qso_redux.utils.utils import open_spectra, wave_0, compute_snr, normalize_spectra
import pandas as pd
from IPython import embed

# -------------------------------------------------------------------------------
# Define the path to the spectra
spectra_dir = "/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/PypeIt_data/REDUX_OUT/spectra_for_plotting_coadd/new_spec"

# Define the path to the quasar list
qso_table = Table.read("/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/qso_list_new_J1243.fits")

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
                spectra_dict[row["qso_name"]].append((filename, row["z"]))
                break

# Sort the dictionary by decreasing redshift
spectra_dict = dict(sorted(spectra_dict.items(), key=lambda x: x[1][0][1], reverse=True))

# Define the wavelength grid
wave_min = 1040.0
wave_max = 3332.0

dvpix =110.0

nqsos= len(spectra_dict.values())
wave_grid = get_wave_grid(wave_min, wave_max, dvpix)
nspec = wave_grid.size

flux_rebin = np.zeros((nqsos, nspec))
ivar_rebin = np.zeros_like(flux_rebin)
gpm_rebin = np.zeros_like(flux_rebin, dtype=bool)
count_rebin = np.zeros_like(flux_rebin)

flux1_rebin = np.zeros((nqsos, nspec))
ivar1_rebin = np.zeros_like(flux1_rebin)
gpm1_rebin = np.zeros_like(flux1_rebin, dtype=bool)
count1_rebin = np.zeros_like(flux1_rebin)

flux2_rebin = np.zeros((nqsos, nspec))
ivar2_rebin = np.zeros_like(flux2_rebin)
gpm2_rebin = np.zeros_like(flux2_rebin, dtype=bool)
count2_rebin = np.zeros_like(flux2_rebin)

# -------------------------------------------------------------------------------
# Creation of the table in which plot the SNR computed in 3 different wavelength ranges
table_data = {'File name': [], 'z': [], 'Mean (J-band)': [], 'Mean (H-band)': [], 'Mean (K-band)': []}

wave_J = [11000., 13400.] # J-band
wave_H = [14500., 17950.] # H-band
wave_K = [19650., 22400.] # K-band

debug = False

for i, spectra_list in enumerate(spectra_dict.values()):

    if len(spectra_list) == 1:
        qso1, z = spectra_list[0]
        wave, flux, ivar, gpm, telluric = open_spectra(spectra_dir, qso1, telluric=True, flux_scale=True)

        wave_rest = wave_0(wave, z)
        flux_rebin[i, :], ivar_rebin[i, :], gpm_rebin[i, :], count_rebin[i, :] = rebin_spectra(wave_grid, wave_rest, flux, ivar, gpm=gpm)
        sigma_rebin = np.sqrt(inverse(ivar_rebin))

        # Compute the SNR in the different wavelength ranges
        wave_J_rest = wave_0(wave_J, z)
        wave_H_rest = wave_0(wave_H, z)
        wave_K_rest = wave_0(wave_K, z)

        result1 = compute_snr(wave_grid, flux_rebin[i,:], ivar_rebin[i,:], gpm_rebin[i, :], wave_J_rest)
        result2 = compute_snr(wave_grid, flux_rebin[i,:], ivar_rebin[i,:], gpm_rebin[i, :], wave_H_rest)
        result3 = compute_snr(wave_grid, flux_rebin[i,:], ivar_rebin[i,:], gpm_rebin[i, :], wave_K_rest)

        # Fill the table
        table_data['File name'].append(qso1[0:10])
        table_data['z'].append(z)
        table_data['Mean (J-band)'].append('{:.1f}'.format(result1[0]))
        table_data['Mean (H-band)'].append('{:.1f}'.format(result2[0]))
        table_data['Mean (K-band)'].append('{:.1f}'.format(result3[0]))

        if debug:
            #Plot the rebinned spectra or the SNR
            snr_flux = flux_rebin[i, :] * inverse(sigma_rebin[i, :]) * gpm_rebin[i, :]
            snr_flux_smooth = fast_running_median(snr_flux, 10)

            fx = plt.figure(figsize=(16, 7))
            ax = fx.add_subplot(111)

            ax.plot(wave_grid, snr_flux, drawstyle='steps-mid', color='darkred', label=qso1[0:10])
            ax.axvspan(wave_J_rest[0], wave_J_rest[1], alpha=0.2, color='navy', label='J-band')
            ax.axvspan(wave_H_rest[0], wave_H_rest[1], alpha=0.2, color='forestgreen', label='H-band')
            ax.axvspan(wave_K_rest[0], wave_K_rest[1], alpha=0.2, color='darkorange', label='K-band')

            if 'XShooter' in qso1:
               ymax = (fast_running_median(flux, 30)).max()
               ymax_xsh = 1.3 * min(ymax, flux_rebin[i, :][(wave_grid > 1215.67 - 20.0) & (wave_grid < 1215.67 + 20.0)].max())
               ax.axis([1085., 3160., -0.02, np.max(snr_flux)+2.0])
            else:
               ymax1 = 1.2 * np.max(flux_rebin[i, :] * gpm_rebin[i, :])
               ax.axis([1085., 3160., -0.02, np.max(snr_flux)+2.0])


            ax.set_xlabel(r'Rest-frame Wavelength [$\rm{\AA}$]', size=22)
            ax.set_ylabel(r'SNR', size=22)
            ax.legend(fontsize=20, loc='upper right')
            ax.tick_params(direction='in', length=6, width=1, which='both', right=True, top=False, labelsize=18)
            plt.tight_layout()
            # plt.savefig('../../figures/SNR_comparison/all/snr_' + qso1[0:10] + '.png', format='png', dpi=500)
            plt.show()
            plt.close()
table = Table(table_data)

# Write a cvs file with the name of the quasar, the redshift and the SNR in the 3 wavelength ranges
new_table = table.copy()
new_table.write('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/snr/SNR_table_mean.csv', format='csv', overwrite=True)

# Read the new csv file to compute the mean of the mean values
df = pd.read_csv('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/snr/SNR_table_mean.csv', sep=',', comment='#', na_values='--')
name = np.array(df['File name'])
z = np.array(df['z'])
mean_j_band = np.array(df['Mean (J-band)'])
mean_h_band = np.array(df['Mean (H-band)'])
mean_k_band = np.array(df['Mean (K-band)'])

# -------------------------------------------------------------------------------
# Plot the histograms of the SNR in the 3 wavelength ranges
mean_j_band = [float(value) for value in table_data['Mean (J-band)']]
mean_h_band = [float(value) for value in table_data['Mean (H-band)']]
mean_k_band = [float(value) for value in table_data['Mean (K-band)']]

bins = np.arange(0, 312, 5) # 5 is the bin width

# Create figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(20, 27), sharex=True)

# J-band histogram
axs[0].hist(mean_j_band, bins=bins, edgecolor='black', alpha=0.5, color='navy', label='J-band', zorder=10)
axs[0].axvline(np.mean(mean_j_band), color='red', linestyle='dashed', linewidth=3, label=r'Mean $\langle \rm{SNR_{J}} \rangle$= %.1f'%np.mean(mean_j_band))
axs[0].axvline(np.median(mean_j_band), color='black', linestyle='dashed', linewidth=3, label=r'Median $\langle \rm{SNR_{J}} \rangle$= %.1f'%np.median(mean_j_band))
axs[0].set_ylabel(r'$\rm{N_{QSOs}}$', size=40)
axs[0].legend(fontsize=40)
axs[0].tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=28)

# H-band histogram
axs[1].hist(mean_h_band, bins=bins, edgecolor='black', alpha=0.5, color='forestgreen', label='H-band', zorder=5)
axs[1].axvline(np.mean(mean_h_band), color='red', linestyle='dashed', linewidth=3, label=r'Mean $\langle \rm{SNR_{H}} \rangle$= %.1f'%np.mean(mean_h_band))
axs[1].axvline(np.median(mean_h_band), color='black', linestyle='dashed', linewidth=3, label=r'Median $\langle \rm{SNR_{H}} \rangle$= %.1f'%np.median(mean_h_band))
axs[1].set_ylabel(r'$\rm{N_{QSOs}}$', size=40)
axs[1].legend(fontsize=40)
axs[1].tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=28)

# K-band histogram
axs[2].hist(mean_k_band, bins=bins, edgecolor='black', alpha=0.5, color='darkorange', label='K-band', zorder=2)
axs[2].axvline(np.mean(mean_k_band), color='red', linestyle='dashed', linewidth=3, label=r'Mean $\langle \rm{SNR_{K}} \rangle$= %.1f'%np.mean(mean_k_band))
axs[2].axvline(np.median(mean_k_band), color='black', linestyle='dashed', linewidth=3, label=r'Median $\langle \rm{SNR_{K}} \rangle$= %.1f'%np.median(mean_k_band))

# Make the x-axis logarithmic
axs[2].set_xscale('log')

axs[2].set_xlabel(r'$\langle \rm{SNR} \rangle$', size=40)
axs[2].set_ylabel(r'$\rm{N_{QSOs}}$', size=40)
axs[2].legend(fontsize=40)
axs[2].tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=28)

plt.tight_layout()
# plt.savefig('../../figures/SNR_comparison/all/histogram_SNR_mean_new_log.png', format='png', dpi=500)
plt.show()
