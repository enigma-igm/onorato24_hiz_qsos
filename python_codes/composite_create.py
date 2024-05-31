# Script to create the composite spectra

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from astropy.table import Table
from pypeit.utils import inverse, fast_running_median
from qso_fitting.data.utils import get_wave_grid, rebin_spectra
from highz_qso_redux.utils.utils import open_spectra, wave_0, flux_0, normalize_spectra

from IPython import embed

# Write some relevant emission lines wavelengths
line_names = [r'Ly$\alpha$', 'NV', 'SiII', 'OI/SiII', 'SiIV/OIV]', 'CIV', 'CIII]', 'MgII']
line_waves = [1215.67, 1240.81, 1260.4221, 1302.168, 1393.76, 1548.19, 1908.734, 2796.35]

# -------------------------------------------------------------------------------
spectra_dir = "/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/PypeIt_data/REDUX_OUT/spectra_for_plotting_coadd/new_spec"

# Define the path to the quasar list
qso_table = Table.read("/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/qso_list_new_J1243_mag.fits")

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
                spectra_dict[row["qso_name"]].append((filename, row["z"], row["M_1450"]))
                break

# Sort the dictionary by decreasing redshift
spectra_dict = dict(sorted(spectra_dict.items(), key=lambda x: x[1][0][1], reverse=True))

# -------------------------------------------------------------------------------
# Define the terms to exclude from the plot
out_sample = []
bals = ['J1243', 'J0313-1806', 'J0038-1527', 'J0839+3900', 'J2348-3054', 'J0246-5219', 'J0430-1445', 'J0910-0414', 'J0923+0402', 'J0706+2921', 'J1526-2050', 'J0439+1634']
excluded_keys = out_sample + bals #Comment if you want to include the BALs
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
wave_min = 1040.0
wave_max = 3332.0
dvpix =110.0 # You don't want it finer than the coarsest pixel scale of one of your instruments in velocity.

nsmooth = int(np.round(10000.0/dvpix)) # This is the number of pixels for the fast-running median filter

nqsos= len(spectra_dict_new.values())
wave_grid_composite = get_wave_grid(wave_min, wave_max, dvpix)
nspec = wave_grid_composite.size
flux_rebin = np.zeros((nqsos, nspec))
ivar_rebin = np.zeros_like(flux_rebin)
gpm_rebin = np.zeros_like(flux_rebin, dtype=bool)
count_rebin = np.zeros_like(flux_rebin)
telluric_rebin = np.zeros_like(flux_rebin)

wave_norm = 1450.0 # Wavelength at which the spectra are normalized
z_array = np.zeros(nqsos)
M_1450_array = np.zeros(nqsos)

debug = False
for i, spectra_list in enumerate(spectra_dict_new.values()):
    qso, z_good, M_1450 = spectra_list[0]
    z_array[i] = z_good
    M_1450_array[i] = M_1450
    wave, flux, ivar, gpm, telluric = open_spectra(spectra_dir, qso, telluric=True, flux_scale=True)
    wave_rest = wave_0(wave, z_good)
    flux_rest, ivar_rest = flux_0(flux, ivar, z_good)
    flux_rebin[i,:], ivar_rebin[i, :], gpm_rebin[i, :], count_rebin[i, :], telluric_rebin[i, :] = rebin_spectra(wave_grid_composite, wave_rest, flux, ivar, gpm=gpm, telluric=telluric)

    flux_rebin[i, :], ivar_rebin[i, :] = normalize_spectra(wave_grid_composite, flux_rebin[i, :], ivar_rebin[i, :], wave_norm)
    sigma_rebin = np.sqrt(inverse(ivar_rebin))

    if debug:
        flux_median = fast_running_median(flux_rebin[gpm_rebin], nsmooth)
        ymax = 1.3*np.max(flux_median)
        ymin = -np.median(sigma_rebin[gpm_rebin])
        fx = plt.figure(1, figsize=(15, 6))
        # left, bottom, width, height
        rect = [0.05, 0.1, 0.9, 0.85]
        ax = fx.add_axes(rect)
        ax.plot(wave_grid_composite, flux_rebin[i, :] * gpm_rebin[i, :], drawstyle='steps-mid', color='black',
                label='rebinned spectrum, z={:5.3f}'.format(z_good[i]), alpha=0.7, zorder=1)
        ax.plot(wave_grid_composite, sigma_rebin[i]*gpm_rebin[i], drawstyle='steps-mid', color='green',
                label='error', alpha=0.7, zorder=1)
        ax.set_ylim((ymin, ymax))
        ax.legend(fontsize=15)
        plt.show()

z_array = np.array(z_array)[:, np.newaxis]
M_1450_array = np.array(M_1450_array)[:, np.newaxis]

# -------------------------------------------------------------------------------
# Divide the sample into 2 redshift bins
z_bin_low = False # True if you want to create the composite spectrum in the low redshift bin (from z<6.70 quasars)
z_bin_high = False # True if you want to create the composite spectrum in the high redshift bin (from z>=6.70 quasars)

z_array_flat = z_array.flatten()

if z_bin_low:
    flux_rebin = flux_rebin[z_array_flat < np.median(z_array_flat)]
    ivar_rebin = ivar_rebin[z_array_flat < np.median(z_array_flat)]
    sigma_rebin = sigma_rebin[z_array_flat < np.median(z_array_flat)]
    gpm_rebin = gpm_rebin[z_array_flat < np.median(z_array_flat)]
    count_rebin = count_rebin[z_array_flat < np.median(z_array_flat)]
    telluric_rebin = telluric_rebin[z_array_flat < np.median(z_array_flat)]
elif z_bin_high:
    flux_rebin = flux_rebin[z_array_flat >= np.median(z_array_flat)]
    ivar_rebin = ivar_rebin[z_array_flat >= np.median(z_array_flat)]
    sigma_rebin = sigma_rebin[z_array_flat >= np.median(z_array_flat)]
    gpm_rebin = gpm_rebin[z_array_flat >= np.median(z_array_flat)]
    count_rebin = count_rebin[z_array_flat >= np.median(z_array_flat)]
    telluric_rebin = telluric_rebin[z_array_flat >= np.median(z_array_flat)]

# -------------------------------------------------------------------------------
# Divide the sample into 2 M_1450 bins
M_1450_bin_bright = False # True if you want to create the composite spectrum in the bright M_1450 bin (M_1450<-26.0)
M_1450_bin_faint = False # True if you want to create the composite spectrum in the faint M_1450 bin (M_1450>=-26.0)

M_1450_array_flat = M_1450_array.flatten()

if M_1450_bin_bright:
    flux_rebin = flux_rebin[M_1450_array_flat < np.median(M_1450_array_flat)]
    ivar_rebin = ivar_rebin[M_1450_array_flat < np.median(M_1450_array_flat)]
    sigma_rebin = sigma_rebin[M_1450_array_flat < np.median(M_1450_array_flat)]
    gpm_rebin = gpm_rebin[M_1450_array_flat < np.median(M_1450_array_flat)]
    count_rebin = count_rebin[M_1450_array_flat < np.median(M_1450_array_flat)]
    telluric_rebin = telluric_rebin[M_1450_array_flat < np.median(M_1450_array_flat)]
elif M_1450_bin_faint:
    flux_rebin = flux_rebin[M_1450_array_flat >= np.median(M_1450_array_flat)]
    ivar_rebin = ivar_rebin[M_1450_array_flat >= np.median(M_1450_array_flat)]
    sigma_rebin = sigma_rebin[M_1450_array_flat >= np.median(M_1450_array_flat)]
    gpm_rebin = gpm_rebin[M_1450_array_flat >= np.median(M_1450_array_flat)]
    count_rebin = count_rebin[M_1450_array_flat >= np.median(M_1450_array_flat)]
    telluric_rebin = telluric_rebin[M_1450_array_flat >= np.median(M_1450_array_flat)]

# -------------------------------------------------------------------------------
# Masking
wave_nomask = 1225.0
lambda_gpm = (wave_grid_composite[np.newaxis,:] < wave_nomask) # To mask out of the Lyman-alpha forest
flux_gpm = (flux_rebin < 40.0) | lambda_gpm # To mask only sky-lines and hot pixels
sigma_gpm = (sigma_rebin < 1.5) | lambda_gpm # To mask very noisy fluxes if the noise is really high
snr_gpm = (flux_rebin / sigma_rebin > 0.5) | lambda_gpm # To mask very low snr fluxes
telluric_gpm = (telluric_rebin > 0.5) | lambda_gpm # To mask out the telluric regions

tot_gpm = flux_gpm & sigma_gpm & snr_gpm & telluric_gpm
# -------------------------------------------------------------------------------
# Total masks
nused = np.sum(gpm_rebin * tot_gpm, axis=0) # This is the number of spectra used at each wavelength
weights = (gpm_rebin * tot_gpm)/nused[np.newaxis,:]
weights_sum = np.sum(weights, axis=0)
norm_weights = weights*inverse(weights_sum)
inv_w_sum = inverse(norm_weights)

flux_comp_gpm = (nused > 0)
flux_comp = flux_comp_gpm*np.sum(flux_rebin*norm_weights, axis=0)
var_comp = flux_comp_gpm*np.sum(sigma_rebin**2*norm_weights**2, axis=0)
sigma_comp = flux_comp_gpm*np.sqrt(var_comp)
# -------------------------------------------------------------------------------
# Find the mean redshift that contributes to each wavelength
if z_bin_low:
    z_array_selected = z_array_flat[z_array_flat < np.median(z_array_flat)]
    z_array_selected_reshaped = z_array_selected.reshape(-1, 1)  # reshape to (16, 1)

    z_mean = np.sum(gpm_rebin * tot_gpm * z_array_selected_reshaped, axis=0) / np.sum(gpm_rebin * tot_gpm, axis=0)
    z_weights0 = (gpm_rebin * tot_gpm * z_array_selected_reshaped) / z_mean[np.newaxis, :]
elif z_bin_high:
    z_array_selected = z_array_flat[z_array_flat >= np.median(z_array_flat)]
    z_array_selected_reshaped = z_array_selected.reshape(-1, 1)  # reshape to (16, 1)

    z_mean = np.sum(gpm_rebin * tot_gpm * z_array_selected_reshaped, axis=0) / np.sum(gpm_rebin * tot_gpm, axis=0)
    z_weights0 = (gpm_rebin * tot_gpm * z_array_selected_reshaped) / z_mean[np.newaxis, :]
# -------------------------------------------------------------------------------
elif M_1450_bin_bright:
    z_array_selected = z_array_flat[M_1450_array_flat < np.median(M_1450_array_flat)]
    z_array_selected_reshaped = z_array_selected.reshape(-1, 1)

    z_mean = np.sum(gpm_rebin * tot_gpm * z_array_selected_reshaped, axis=0) / np.sum(gpm_rebin * tot_gpm, axis=0)
    z_weights0 = (gpm_rebin * tot_gpm * z_array_selected_reshaped) / z_mean[np.newaxis, :]
elif M_1450_bin_faint:
    z_array_selected = z_array_flat[M_1450_array_flat >= np.median(M_1450_array_flat)]
    z_array_selected_reshaped = z_array_selected.reshape(-1, 1)

    z_mean = np.sum(gpm_rebin * tot_gpm * z_array_selected_reshaped, axis=0) / np.sum(gpm_rebin * tot_gpm, axis=0)
    z_weights0 = (gpm_rebin * tot_gpm * z_array_selected_reshaped) / z_mean[np.newaxis, :]
# -------------------------------------------------------------------------------
else:
    z_mean = np.sum(gpm_rebin * tot_gpm * z_array, axis=0) / np.sum(gpm_rebin * tot_gpm, axis=0) # This is the mean redshift at each wavelength
    z_weights0 = (gpm_rebin * tot_gpm * z_array)/z_mean[np.newaxis,:]
# -------------------------------------------------------------------------------
# Save the composite spectrum
save = False
if save:
    # Save the composite spectrum
    composite_fits = {'Wavelengths': [], 'Flux': [], 'Error': [], 'Nspec': [], 'Mean_z': []}
    composite_fits['Wavelengths'] = wave_grid_composite
    composite_fits['Flux'] = flux_comp
    composite_fits['Error'] = sigma_comp
    composite_fits['Nspec'] = nused
    composite_fits['Mean_z'] = z_mean
    save_comp = Table(composite_fits)
    save_comp.write('../../tables/composite/composite_Onorato_dv110.fits', overwrite=True) #Change the name to save the different versions

# -------------------------------------------------------------------------------
# Plot the composite spectrum, N_spectra used at each pixel, and the mean redshift at each pixel
flux_median1 = fast_running_median(flux_comp, nsmooth)
ymax1 = 3.1 * np.max(flux_median1)
ymin1 = -np.median(sigma_comp)

ymax_xsh = min(ymax1, 1.5 * flux_comp[(wave_grid_composite > 1215.67 - 20.0) & (wave_grid_composite < 1215.67 + 20.0)].max())

fx = plt.figure(figsize=(17, 9))
gs = gridspec.GridSpec(12, 5, figure=fx, wspace=0., hspace=0.)
ax = fx.add_subplot(gs[4:12, 0:5])
ax1 = fx.add_subplot(gs[2:4, 0:5], sharex=ax)
ax2 = fx.add_subplot(gs[0:2, 0:5], sharex=ax)

ax.axvspan(min(wave_grid_composite), wave_nomask, alpha=0.3, color='dimgrey', label=r'$\lambda_{NOmasks}<$' + r'{:.0f}'.format(wave_nomask) + ' ' + r'$\rm{\AA}$')
ax1.axvspan(min(wave_grid_composite), wave_nomask, alpha=0.3, color='dimgrey', label='mask-free region')
ax1.plot(wave_grid_composite, nused, drawstyle='steps-mid', color='black', label= 'tell,SNR,$\sigma$,flux masks', alpha=0.8, zorder=11)
ax2.plot(wave_grid_composite, z_mean, drawstyle='steps-mid', color='black', label= '$z_{mean}$', alpha=0.8, zorder=11)
ax.plot(wave_grid_composite, flux_comp, drawstyle='steps-mid', color='black', label='Onorato+24 (this work)', alpha=0.8, zorder=11)
ax.plot(wave_grid_composite, sigma_comp, drawstyle='steps-mid', color='gray', alpha=0.7, zorder=11)

for pos, name in zip(line_waves, line_names):
    ax.axvline(x=pos, color='navy', linestyle='--', linewidth=0.75)
    ax.annotate(name, xy=(pos, 4.5), xytext=(-10.5, 20), textcoords='offset points', rotation=90, fontsize=10)

# -------------------------------------------------------------------------------
# EXQR-30 total composite
exqr30 = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/Composites_lit/EXQR-30tot.txt'
wave_exqr30, flux_exqr30 = np.genfromtxt(exqr30, skip_header=(2), unpack=True)
ivar_exqr30 = np.ones_like(flux_exqr30) # Fake ivar
flux_exqr30_norm, ivar_exqr30_norm = normalize_spectra(wave_exqr30, flux_exqr30, ivar_exqr30, wave_norm)
ax.plot(wave_exqr30, flux_exqr30_norm, drawstyle='steps-mid', color='red', label="D'Odorico+23", alpha=0.7, zorder=10)
# -------------------------------------------------------------------------------
# Yang composite
yang = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/Composites_lit/Yang2021.dat'
wave_yang, flux_yang, nobj_yang = np.genfromtxt(yang, skip_header=(17), unpack=True)
ivar_yang = np.ones_like(flux_yang) # Fake ivar
flux_yang_norm, ivar_yang_norm = normalize_spectra(wave_yang, flux_yang, ivar_yang, wave_norm)
ax.plot(wave_yang, flux_yang_norm, drawstyle='steps-mid', color='darkorange', label='Yang+21', alpha=0.7, zorder=9)
ax1.plot(wave_yang, nobj_yang, drawstyle='steps-mid', color='darkorange', label= 'Yang+21', alpha=0.7, zorder=9)
# -------------------------------------------------------------------------------
# Shen composite
shen = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/Composites_lit/Shen2019.dat'
wave_shen, flux_shen, err_shen, nobj_shen = np.genfromtxt(shen, skip_header=(23), unpack=True)
ivar_shen = err_shen**(-2)
flux_shen_norm, ivar_shen_norm = normalize_spectra(wave_shen, flux_shen, ivar_shen, wave_norm)
ax.plot(wave_shen, flux_shen_norm, drawstyle='steps-mid', color='forestgreen', label='Shen+19', alpha=0.7, zorder=8)
ax1.plot(wave_shen, nobj_shen, drawstyle='steps-mid', color='forestgreen', label= 'Shen+19', alpha=0.7, zorder=8)
# -------------------------------------------------------------------------------
# Selsing composite
selsing = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/Composites_lit/Selsing2015.dat'
wave_sels, flux_sels, err_sels = np.genfromtxt(selsing, skip_header=(1), unpack=True)
ivar_sels = err_sels**(-2)
flux_sels_norm, ivar_sels_norm = normalize_spectra(wave_sels, flux_sels, ivar_sels, wave_norm)
ax.plot(wave_sels, flux_sels_norm, drawstyle='steps-mid', color='deepskyblue', label='Selsing+16', alpha=0.7, zorder=7)
# -------------------------------------------------------------------------------
# Telfer composite
telfer = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/Composites_lit/Telfer2002.dat'
wave_telf, flux_telf, err_telf = np.genfromtxt(telfer, skip_header=(2), unpack=True)
ivar_telf = err_telf**(-2)
flux_telf_norm, ivar_telf_norm = normalize_spectra(wave_telf, flux_telf, ivar_telf, wave_norm)
ax.plot(wave_telf, flux_telf_norm, drawstyle='steps-mid', color='blue', label='Telfer+02', alpha=0.7, zorder=6)
# -------------------------------------------------------------------------------
# Vanden Berk composite
vandenberk = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/Composites_lit/VandenBerk2001.dat'
wave_vand, flux_vand, err_vand = np.genfromtxt(vandenberk, skip_header=(23), unpack=True)
ivar_vand = err_vand**(-2)
flux_vand_norm, ivar_vand_norm = normalize_spectra(wave_vand, flux_vand, ivar_vand, wave_norm)
ax.plot(wave_vand, flux_vand_norm, drawstyle='steps-mid', color='purple', label='Vanden Berk+01', alpha=0.7, zorder=5)
# -------------------------------------------------------------------------------
# Inset panel with a zoom
axins = ax.inset_axes([0.4, 0.25, 0.35, 0.74]) #left, bottom, width, height
axins.plot(wave_grid_composite, flux_comp, drawstyle='steps-mid', color='black', label='Onorato+24 (this work)', alpha=0.8, zorder=11)
axins.plot(wave_grid_composite, sigma_comp, drawstyle='steps-mid', color='gray', alpha=0.7, zorder=11)
axins.plot(wave_exqr30, flux_exqr30_norm, drawstyle='steps-mid', color='red', label='E-XQR-30', alpha=0.7, zorder=10)
axins.plot(wave_yang, flux_yang_norm, drawstyle='steps-mid', color='orange', label='Yang+21', alpha=0.7, zorder=9)
axins.plot(wave_shen, flux_shen_norm, drawstyle='steps-mid', color='green', label='Shen+19', alpha=0.7, zorder=8)
axins.plot(wave_sels, flux_sels_norm, drawstyle='steps-mid', color='deepskyblue', label='Selsing+16', alpha=0.7, zorder=7)
axins.plot(wave_telf, flux_telf_norm, drawstyle='steps-mid', color='blue', label='Telfer+02', alpha=0.7, zorder=6)
axins.plot(wave_vand, flux_vand_norm, drawstyle='steps-mid', color='purple', label='Vanden Berk+01', alpha=0.7, zorder=5)
axins.set_xlim(1175., 1580.)
axins.set_ylim(-0.05, 5.01)
axins.set_xticks([1200.,1300.,1400.,1500.])
axins.set_yticks([0.,1.,2.,3.,4.])
axins.set_facecolor('whitesmoke')
axins.tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=17)
# -------------------------------------------------------------------------------

ax.axis([1038.,3338.,-0.05,5.4])
ax.set_xlabel(r'Rest-frame Wavelength [$\rm{\AA}$]', size=23)
ax.set_ylabel(r'Flux', size=23)
ax.legend(fontsize=15, loc='upper right')
ax.tick_params(direction='in', length=6, width=1, which='both', right=True, top=False, labelsize=20)

ax1.set_ylabel(r'$\rm{N_{spectra}}$', size=23)
ax1.set_yticks([0, 20, 45])
ax1.axis([1038.,3338.,-0.05,52.])
ax1.tick_params(direction='in', length=6, width=1, which='both', right=True, top=False, labelsize=20)

ax2.set_ylabel(r'$\rm{z_{mean}}$', size=23)
ax2.set_yticks([6.6,6.9,7.2])
ax2.axis([1038.,3338.,6.49,7.3])
ax2.tick_params(direction='in', length=6, width=1, which='both', right=True, top=False, labelsize=20)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.tight_layout()
# plt.savefig('../../figures/composite_spectrum_new_1450.png', bbox_inches='tight', dpi=500)
plt.show()

# -------------------------------------------------------------------------------
# Plot the composite spectrum and all the individual spectra in list_qsos in a single plot as a check of the composite

cols = sns.color_palette("nipy_spectral", len(spectra_dict_new.values()))

fx = plt.figure(figsize=(17, 9.5))
ax = fx.add_subplot(111)
ax.plot(wave_grid_composite, flux_comp, drawstyle='steps-mid', color='black', label='Composite', alpha=0.9, zorder=5)
for i, qso in enumerate(spectra_dict_new.values()):
    ax.plot(wave_grid_composite, flux_rebin[i, :], drawstyle='steps-mid', color = cols[i], label=f"{qso[0][0][0:10]} z={qso[0][1]:.3f}", alpha=0.4, zorder=4)

for pos, name in zip(line_waves, line_names):
    plt.axvline(x=pos, color='navy', linestyle='--', zorder=6)
    plt.annotate(name, xy=(pos, 4.5), xytext=(-10.5, 20), textcoords='offset points', rotation=90, zorder=6)

ax.axis([1038.,3330.,-0.05,5.4])
ax.set_xlabel(r'Rest-frame Wavelength [$\rm{\AA}$]', size=23)
ax.set_ylabel(r'Flux', size=23)
ax.legend(fontsize=8.75, loc='upper right')
ax.tick_params(direction='in', length=6, width=1, which='both', right=True, top=False, labelsize=20)

plt.tight_layout()
# plt.savefig('../../figures/composite_chk_conf.pdf', bbox_inches='tight', dpi=500)
plt.show()