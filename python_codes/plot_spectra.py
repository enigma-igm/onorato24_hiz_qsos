# Script to plot the spectra of the quasars

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import seaborn as sns
from astropy.table import Table
import astropy.cosmology as cosmo
from highz_qso_redux.utils import utils
from pypeit.utils import inverse, fast_running_median
from highz_qso_redux.utils.utils import open_spectra
from highz_qso_redux.utils.color_map import tol_cmap
from qso_fitting.data.fluxing.flux_correct import spec_interp_gpm
from IPython import embed

planck = cosmo.Planck18

# -------------------------------------------------------------------------------
# plotting settings
lsize = 30
lsize_text = 28
matplotlib.rcParams['axes.grid'] = False
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "pale red", "orange"]
colors = sns.xkcd_palette(colors)
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'axes.edgecolor':'white'})
sns.set_style("ticks")
matplotlib.rcParams['ytick.labelsize'] = lsize_text
matplotlib.rcParams['xtick.labelsize'] = lsize_text
hfont = {'fontname':'Arial'}

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

# -------------------------------------------------------------------------------
# Define the terms to exclude from the plot
bals = ['J1243', 'J0313-1806', 'J0038-1527', 'J0839+3900', 'J2348-3054', 'J0246-5219', 'J0430-1445', 'J0910-0414', 'J0923+0402', 'J0706+2921', 'J1526-2050', 'J0439+1634']
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

# -------------------------------------------------------------------------------
# Split the spectra into three groups to plot them in three different figures
spectra_list1 = list(spectra_dict_new.values())[:len(spectra_dict_new.values())//3]
spectra_list2 = list(spectra_dict_new.values())[len(spectra_dict_new.values())//3:2*len(spectra_dict_new.values())//3]
spectra_list3 = list(spectra_dict_new.values())[2*len(spectra_dict_new.values())//3:]

n_spectra = len(list(spectra_dict_new.values()))
n_spectra1 = len(spectra_list1)
n_spectra2 = len(spectra_list2)
n_spectra3 = len(spectra_list3)
tot_colors = n_spectra1 + n_spectra2 + n_spectra3

color_palette = tol_cmap('rainbow_PuBr', tot_colors)
color_palette = color_palette.reversed() # Reverse the color palette

color_group1 = [color_palette(i / tot_colors) for i in range(n_spectra1)]
color_group2 = [color_palette(i / tot_colors) for i in range(n_spectra1, n_spectra1 + n_spectra2)]
color_group3 = [color_palette(i / tot_colors) for i in range(n_spectra1 + n_spectra2, tot_colors)]

# Plot the first group of spectra
fig, ax = plt.subplots(n_spectra1, 1, figsize=(16, 22), sharex=True)
x_min = 8245.
x_max = 22980.
flux_drop = 0.03

tel_band1 = [13500., 14150.]
tel_band2 = [18200., 19300.]

to_cut = ['J0410']

# List of quasars to smooth with a different window
tosmooth = ['J1243', 'J0910-', 'J2338', 'J1110', 'J0109-', 'J2348', 'J1048']

window = 10
window_smooth = 20

for i, spectra_list in enumerate(spectra_list1):
    qso, z = spectra_list[0]
    wave, flux, ivar, gpm, telluric = open_spectra(spectra_dir, qso, telluric=True, flux_scale=True)
    flux_interp, ivar_interp = spec_interp_gpm(wave, flux, gpm, sigma_or_ivar=ivar)

    flux_sm, ivar_sm = utils.ivarsmooth(flux_interp, ivar_interp, window=window)
    gpm_sm = (ivar_sm > 0.0)
    sigma_sm = np.sqrt(inverse(ivar_sm))

    ymax = (fast_running_median(flux_sm, 30)).max()
    ymax1 = 1.1 * max(flux_sm * gpm_sm)
    ymax_xsh = 1.3 * min(ymax, flux_sm[(wave > 1215.67 * (1 + z) - 20.0) & (wave < 1215.67 * (1 + z) + 20.0)].max())
    y_min = -0.1 * min(ymax, flux_sm[(wave > 1215.67 * (1 + z) - 20.0) & (wave < 1215.67 * (1 + z) + 20.0)].max())

    if any(term in qso for term in tosmooth):
        flux_sm, ivar_sm = utils.ivarsmooth(flux_interp, ivar_interp, window=window_smooth)
        gpm_sm = (ivar_sm > 0.0)
        sigma_sm = np.sqrt(inverse(ivar_sm))

    # Mask the bad pixels
    mask_bad3 = np.logical_and(wave > 19300., wave < np.max(wave))
    mask_bad2 = np.logical_and(wave > 14150., wave < 18200.)
    mask_bad1 = np.logical_and(wave > np.min(wave), wave < 13500.)
    mask_tot = np.logical_or(mask_bad1, mask_bad2) | np.logical_or(mask_bad2, mask_bad3)
    flux_sm[(~mask_bad1) & (~mask_bad2) & (~mask_bad3)] = np.nan
    sigma_sm[(~mask_bad1) & (~mask_bad2) & (~mask_bad3)] = np.nan

    ax[i].axvspan(tel_band1[0], tel_band1[1], alpha=0.2, color='darkgray')
    ax[i].axvspan(tel_band2[0], tel_band2[1], alpha=0.2, color='darkgray')

    if 'J1342' in qso:
        cut_flux = 10100.
        mask_bad = (wave > cut_flux)
        flux_sm[~mask_bad] = np.nan
        sigma_sm[~mask_bad] = np.nan

    elif any(term in qso for term in to_cut):
        cut_flux_keck = 9610.
        mask_bad_keck = (wave > cut_flux_keck)
        flux_sm[~mask_bad_keck] = np.nan
        sigma_sm[~mask_bad_keck] = np.nan

    ax[i].tick_params(axis=u'both', direction='in', which='both', length=6, width=1, right=True, top=True, labelsize=lsize_text)
    ax[i].plot(wave, sigma_sm, color=colors[2], lw=2, drawstyle='steps-mid')
    color = color_group1[i % len(color_group1)]  # Use modulo to avoid index out of range
    ax[i].plot(wave, flux_sm, lw=2, drawstyle='steps-mid', color=color)

    # Check if the quasar is in the "bals" list
    if any(bal in qso for bal in bals):
        box_color = 'antiquewhite'
    else:
        box_color = 'white'

    ax[i].text(0.78, 0.75, f"{qso[0:10]} z={z:.4f}", transform=ax[i].transAxes, fontsize=18,
               bbox=dict(facecolor=box_color, alpha=0.8, boxstyle='round', edgecolor='silver', pad=0.15))

    ax[i].set_yticks(np.linspace(0.0, ymax_xsh/1.4, 2))
    ax[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

    if 'XShooter' in qso:
        ax[i].axis([x_min, x_max, y_min, ymax_xsh])
        if 'J1048' in qso:
            ax[i].axis([x_min, x_max, y_min, 1.6])
    elif 'J2338' in qso:
        ax[i].axis([x_min,x_max, y_min, 0.8])
    elif 'J1243' in qso:
        ax[i].axis([x_min, x_max, -0.01, 0.11])
    else:
        ax[i].axis([x_min, x_max, y_min, ymax1])

fig.text(0.05, 0.5, r'Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]', va='center', rotation='vertical', fontsize=lsize, **hfont)
ax[-1].set_xlabel(r'Observed Wavelength [$\rm{\AA}$]', fontsize=lsize, **hfont)
plt.subplots_adjust(hspace=0)
# plt.savefig('../../figures/spectra_group1n.png', bbox_inches='tight', dpi=200)
plt.show()

# Plot the second group of spectra
fig, ax = plt.subplots(n_spectra2, 1, figsize=(16, 22), sharex=True)

for i, spectra_list in enumerate(spectra_list2):
    qso, z = spectra_list[0]
    wave, flux, ivar, gpm, telluric = open_spectra(spectra_dir, qso, telluric=True, flux_scale=True)
    flux_interp, ivar_interp = spec_interp_gpm(wave, flux, gpm, sigma_or_ivar=ivar)

    flux_sm, ivar_sm = utils.ivarsmooth(flux_interp, ivar_interp, window=window)
    gpm_sm = (ivar_sm > 0.0)
    sigma_sm = np.sqrt(inverse(ivar_sm))

    ymax = (fast_running_median(flux_sm, 30)).max()
    ymax1 = 1.1 * max(flux_sm * gpm_sm)
    ymax_xsh = 1.3 * min(ymax, flux_sm[(wave > 1215.67 * (1 + z) - 20.0) & (wave < 1215.67 * (1 + z) + 20.0)].max())
    y_min = -0.1 * min(ymax, flux_sm[(wave > 1215.67 * (1 + z) - 20.0) & (wave < 1215.67 * (1 + z) + 20.0)].max())

    if any(term in qso for term in tosmooth):
        flux_sm, ivar_sm = utils.ivarsmooth(flux_interp, ivar_interp, window=window_smooth)
        gpm_sm = (ivar_sm > 0.0)
        sigma_sm = np.sqrt(inverse(ivar_sm))

    # Mask the bad pixels
    mask_bad3 = np.logical_and(wave > 19300., wave < np.max(wave))
    mask_bad2 = np.logical_and(wave > 14150., wave < 18200.)
    mask_bad1 = np.logical_and(wave > np.min(wave), wave < 13500.)
    mask_tot = np.logical_or(mask_bad1, mask_bad2) | np.logical_or(mask_bad2, mask_bad3)
    flux_sm[(~mask_bad1) & (~mask_bad2) & (~mask_bad3)] = np.nan
    sigma_sm[(~mask_bad1) & (~mask_bad2) & (~mask_bad3)] = np.nan

    ax[i].axvspan(tel_band1[0], tel_band1[1], alpha=0.2, color='darkgray')
    ax[i].axvspan(tel_band2[0], tel_band2[1], alpha=0.2, color='darkgray')

    if 'J1342' in qso:
        cut_flux = 10100.
        mask_bad = (wave > cut_flux)
        flux_sm[~mask_bad] = np.nan
        sigma_sm[~mask_bad] = np.nan

    elif any(term in qso for term in to_cut):
        cut_flux_keck = 9610.
        mask_bad_keck = (wave > cut_flux_keck)
        flux_sm[~mask_bad_keck] = np.nan
        sigma_sm[~mask_bad_keck] = np.nan

    ax[i].tick_params(axis=u'both', direction='in', which='both', length=6, width=1, right=True, top=True, labelsize=lsize_text)
    ax[i].plot(wave, sigma_sm, color=colors[2], lw=2, drawstyle='steps-mid')
    color = color_group2[i % len(color_group2)]  # Use modulo to avoid index out of range
    ax[i].plot(wave, flux_sm, lw=2, drawstyle='steps-mid', color=color)

    # Check if the quasar is in the "bals" list
    if any(bal in qso for bal in bals):
        box_color = 'antiquewhite'
    else:
        box_color = 'white'

    ax[i].text(0.78, 0.75, f"{qso[0:10]} z={z:.4f}", transform=ax[i].transAxes, fontsize=18,
               bbox=dict(facecolor=box_color, alpha=0.8, boxstyle='round', edgecolor='silver', pad=0.15))

    if 'J1917' in qso or 'J2002' in qso:
        ax[i].set_yticks(np.linspace(0.0, ymax1 / 1.1-0.6, 2))
    elif 'J1048' in qso:
        ax[i].set_yticks(np.linspace(0.0, 1.0, 2))
    else:
        ax[i].set_yticks(np.linspace(0.0, ymax_xsh / 1.4, 2))
    ax[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

    if 'XShooter' in qso:
        ax[i].axis([x_min, x_max, y_min, ymax_xsh])
        if 'J1048' in qso:
            ax[i].axis([x_min, x_max, y_min, 1.6])
    elif 'J2338' in qso:
        ax[i].axis([x_min, x_max, y_min, 0.8])
    elif 'J0218' in qso:
        ax[i].axis([x_min, x_max, y_min, 1.01])
    else:
        ax[i].axis([x_min, x_max, y_min, ymax1])

fig.text(0.05, 0.5, r'Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]', va='center', rotation='vertical',
             fontsize=lsize, **hfont)
ax[-1].set_xlabel(r'Observed Wavelength [$\rm{\AA}$]', fontsize=lsize, **hfont)
plt.subplots_adjust(hspace=0)
# plt.savefig('../../figures/spectra_group2n.png', bbox_inches='tight', dpi=200)
plt.show()

# Plot the third group of spectra
fig, ax = plt.subplots(n_spectra3, 1, figsize=(16, 22), sharex=True)

for i, spectra_list in enumerate(spectra_list3):

    qso, z = spectra_list[0]
    wave, flux, ivar, gpm, telluric = open_spectra(spectra_dir, qso, telluric=True, flux_scale=True)
    flux_interp, ivar_interp = spec_interp_gpm(wave, flux, gpm, sigma_or_ivar=ivar)

    flux_sm, ivar_sm = utils.ivarsmooth(flux_interp, ivar_interp, window=window)
    gpm_sm = (ivar_sm > 0.0)
    sigma_sm = np.sqrt(inverse(ivar_sm))

    ymax = (fast_running_median(flux_sm, 30)).max()
    ymax1 = 1.1 * max(flux_sm * gpm_sm)
    ymax_xsh = 1.3 * min(ymax, flux_sm[(wave > 1215.67 * (1 + z) - 20.0) & (wave < 1215.67 * (1 + z) + 20.0)].max())
    y_min = -0.1 * min(ymax, flux_sm[(wave > 1215.67 * (1 + z) - 20.0) & (wave < 1215.67 * (1 + z) + 20.0)].max())

    if any(term in qso for term in tosmooth):
        flux_sm, ivar_sm = utils.ivarsmooth(flux_interp, ivar_interp, window=window_smooth)
        gpm_sm = (ivar_sm > 0.0)
        sigma_sm = np.sqrt(inverse(ivar_sm))

    # Mask the bad pixels
    mask_bad3 = np.logical_and(wave > 19350., wave < np.max(wave))
    mask_bad2 = np.logical_and(wave > 14150., wave < 18200.)
    mask_bad1 = np.logical_and(wave > np.min(wave), wave < 13500.)
    mask_tot = np.logical_or(mask_bad1, mask_bad2) | np.logical_or(mask_bad2, mask_bad3)
    flux_sm[(~mask_bad1) & (~mask_bad2) & (~mask_bad3)] = np.nan
    sigma_sm[(~mask_bad1) & (~mask_bad2) & (~mask_bad3)] = np.nan

    ax[i].axvspan(tel_band1[0], tel_band1[1], alpha=0.2, color='darkgray')
    ax[i].axvspan(tel_band2[0], tel_band2[1], alpha=0.2, color='darkgray')

    if 'J1342' in qso:
        cut_flux = 10100.
        mask_bad = (wave > cut_flux)
        flux_sm[~mask_bad] = np.nan
        sigma_sm[~mask_bad] = np.nan

    elif any(term in qso for term in to_cut):
        cut_flux_keck = 9610.
        mask_bad_keck = (wave > cut_flux_keck)
        flux_sm[~mask_bad_keck] = np.nan
        sigma_sm[~mask_bad_keck] = np.nan

    ax[i].tick_params(axis=u'both', direction='in', which='both', length=6, width=1, right=True, top=True, labelsize=lsize_text)
    ax[i].plot(wave, sigma_sm, color=colors[2], lw=2, drawstyle='steps-mid')
    color = color_group3[i % len(color_group3)]  # Use modulo to avoid index out of range
    ax[i].plot(wave, flux_sm, lw=2, drawstyle='steps-mid', color=color)

    # Check if the quasar is in the "bals" list
    if any(bal in qso for bal in bals):
        box_color = 'antiquewhite'
    else:
        box_color = 'white'

    ax[i].text(0.78, 0.75, f"{qso[0:10]} z={z:.4f}", transform=ax[i].transAxes, fontsize=18,
               bbox=dict(facecolor=box_color, alpha=0.8, boxstyle='round', edgecolor='silver', pad=0.15))

    if 'J0706' in qso:
        ax[i].set_yticks(np.linspace(0.0, ymax1 / 1.1-1.0, 2))
    elif 'J0910-' in qso or 'J2338' in qso:
        ax[i].set_yticks(np.linspace(0.0, 0.5, 2))
    else:
        ax[i].set_yticks(np.linspace(0.0, ymax_xsh / 1.4, 2))
    ax[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

    if 'XShooter' in qso:
        ax[i].axis([x_min, x_max, y_min, ymax_xsh])
        if 'J1048' in qso:
            ax[i].axis([x_min, x_max, y_min, 1.6])
    elif 'J2338' in qso:
        ax[i].axis([x_min, x_max, y_min, 0.7])
    elif 'J0910-' in qso:
        ax[i].axis([x_min, x_max, y_min, 0.85])
    else:
        ax[i].axis([x_min, x_max, y_min, ymax1])

fig.text(0.05, 0.5, r'Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]', va='center', rotation='vertical',
         fontsize=lsize, **hfont)
ax[-1].set_xlabel(r'Observed Wavelength [$\rm{\AA}$]', fontsize=lsize, **hfont)
plt.subplots_adjust(hspace=0)
# plt.savefig('../../figures/spectra_group3n.png', bbox_inches='tight', dpi=200)
plt.show()
