# Code to plot the M1450 vs z, J-band magnitude vs z and the scatter plot of the M1450 from the spectrum vs the M1450 from the photometry

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython import embed
import pandas as pd
from highz_qso_redux.utils.color_map import tol_cmap

# J0910 and J0923 are BALs and show features in the spectrum that make the M1450 unreliable, so I use the inferred values from the bal_continuum.py script
# J1243 is poor quality data and I don't trust M1450 from the spectrum, so I use the tabulated one from the paper

bals = ['J091054', 'J092347']
bad_mag = ['J1243']
replace_mag = bals + bad_mag

# -------------------------------------------------------------------------------
# Read the table with the magnitudes
cat_mag = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/magnitudes/J_mag_list_M1450.csv'
df = pd.read_csv(cat_mag, sep=',', comment='#', na_values='--')

name = np.array(df['qso_name'])
z = np.array(df['z'])
err_z = np.array(df['err_z'])
mag_J = np.array(df['J_ab'])
err_J = np.array(df['err_J'])
passband = np.array(df['passband'])
keyword = np.array(df['keyword'])
M1450_tab = np.array(df['M1450_tab'])
mag_Y = np.array(df['mag_Y'])
err_Y = np.array(df['err_Y'])
mag_Kp = np.array(df['mag_Kp'])
err_Kp = np.array(df['err_Kp'])
M_1450_spec = np.array(df['M_1450_spec'])
M_1450_phot = np.array(df['M_1450_phot'])
name_short = np.array(df['short_name'])

color_palette = tol_cmap('rainbow_PuBr', len(name))
color_palette = color_palette.reversed() # Reverse the color palette

color_group = [color_palette(i / len(name)) for i in range(len(name))]

# -------------------------------------------------------------------------------
# Create the plot
fig = plt.figure(figsize=(15,26))

gs = GridSpec(11, 6, figure=fig)
gs.update(wspace=0., hspace=0.)
ax0 = fig.add_subplot(gs[1:6, 0:5]) # J-band magnitude vs z
ax1 = fig.add_subplot(gs[6:11, 0:5]) # M1450 vs z
ax2 = fig.add_subplot(gs[0, 0:5], sharex=ax0) # Inset panel with a histogram of the redshifts
ax3 = fig.add_subplot(gs[6:11, 5], sharey=ax1) # Inset panel with a histogram of the M1450

for i, qso in enumerate(name):
    if 'J1243' in qso :
        cols = color_group[i % len(color_group)]  # Use modulo to avoid index out of range
        ax0.errorbar(z[i], mag_Y[i], xerr=err_z[i], yerr=err_Y[i], capsize=7, elinewidth=1, capthick=1, linestyle='', marker='*', markersize=20, color=cols, label=f"{name_short[i]}", zorder=10, fillstyle='none')
    elif '092358' in qso:
        cols = color_group[i % len(color_group)]  # Use modulo to avoid index out of range
        ax0.errorbar(z[i], mag_Y[i], xerr=err_z[i], yerr=err_Y[i], capsize=7, elinewidth=1, capthick=1, linestyle='', marker='o', markersize=13, color=cols, label=f"{name_short[i]}", zorder=10, fillstyle='none')
    elif 'J1058' in qso:
        cols = color_group[i % len(color_group)]  # Use modulo to avoid index out of range
        ax0.errorbar(z[i], mag_Kp[i], xerr=err_z[i], yerr=err_Kp[i], capsize=7, elinewidth=1, capthick=1, linestyle='', marker='o', markersize=13, color=cols, label=f"{name_short[i]}", zorder=10, fillstyle='none')
    elif any(term in qso for term in bals):
        cols = color_group[i % len(color_group)]  # Use modulo to avoid index out of range
        ax0.errorbar(z[i], mag_J[i], xerr=err_z[i], yerr=err_J[i], capsize=7, elinewidth=1, capthick=1, linestyle='', marker='*', markersize=20, color=cols, label=f"{name_short[i]}", zorder=10)
    else:
        cols = color_group[i % len(color_group)]  # Use modulo to avoid index out of range
        ax0.errorbar(z[i], mag_J[i], xerr=err_z[i], yerr=err_J[i], capsize=7, elinewidth=1, capthick=1, linestyle='', marker='o', markersize=13, color=cols, label=f"{name_short[i]}", zorder=10)

ax0.set_ylabel(r'J-band magnitude', size=30)
ax0.axis([6.48,7.68,23.9,17.2])
ax0.set_xticks([6.50, 6.75, 7.00, 7.25, 7.50])
ax0.tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=28, labelbottom=False)
ax0.text(7.55, 23.4, r'(a)', color='black', size=30)

# Get handles and labels from the plot
handles, labels = ax0.get_legend_handles_labels()
# Calculate the midpoint
midpoint = len(handles) // 2
# Create two separate legends
ax0.legend(handles[:midpoint], labels[:midpoint], fontsize=21, loc='upper right', ncol=3, frameon=True)

# Histogram of the redshifts
bins = np.arange(6.5, 7.65, 0.05) # 0.05 is the bin width
ax2.hist(z, bins=bins, edgecolor='black', alpha=0.8, color='teal')
ax2.axvline(np.median(z), color='darkred', linestyle='--', linewidth=3, label=r'$z_{\rm{median}}$')
ax2.set_yticks([0, 3, 6])
ax2.get_xaxis().set_visible(False)
ax2.set_xlim(6.48, 7.68)
ax2.text(7.55, 4.5, r'(c)', color='black', size=30)
ax2.tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=28)
ax2.set_ylabel(r'$\rm{N_{QSOs}}$', size=30)

# -------------------------------------------------------------------------------
M1450_plot = np.zeros_like(M_1450_spec)
for i, qso in enumerate(name):
    if any(term in qso for term in bad_mag):
        M1450_plot[i] = M1450_tab[i]
        cols = color_group[i % len(color_group)]  # Use modulo to avoid index out of range
        ax1.errorbar(z[i], M1450_tab[i], xerr=err_z[i], capsize=7, elinewidth=1, capthick=1, linestyle='', marker='*', markersize=20, color=cols, label=f"{name_short[i]}", zorder=10, fillstyle='none')
    elif any(term in qso for term in bals):
        M1450_plot[i] = M_1450_spec[i]
        cols = color_group[i % len(color_group)]  # Use modulo to avoid index out of range
        ax1.errorbar(z[i], M_1450_spec[i], xerr=err_z[i], capsize=7, elinewidth=1, capthick=1, linestyle='', marker='*', markersize=20, color=cols, label=f"{name_short[i]}", zorder=10)
    elif '092358' in qso or 'J1058' in qso:
        M1450_plot[i] = M_1450_spec[i]
        cols = color_group[i % len(color_group)]  # Use modulo to avoid index out of range
        ax1.errorbar(z[i], M_1450_spec[i], xerr=err_z[i], capsize=7, elinewidth=1, capthick=1, linestyle='', marker='o', markersize=13, color=cols, label=f"{name_short[i]}", zorder=10, fillstyle='none')
    else:
        M1450_plot[i] = M_1450_spec[i]
        cols = color_group[i % len(color_group)]  # Use modulo to avoid index out of range
        ax1.errorbar(z[i], M_1450_spec[i], xerr=err_z[i], capsize=7, elinewidth=1, capthick=1, linestyle='', marker='o', markersize=13, color=cols, label=f"{name_short[i]}", zorder=10)

ax1.set_xlabel(r'Redshift', size=30)
ax1.set_ylabel(r'M$_{1450}$', size=30)
ax1.axis([6.48,7.68,-23.9,-29.1])#-24.9,-29.7])
ax1.set_xticks([6.50, 6.75, 7.00, 7.25, 7.50])
ax1.tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=28)
ax1.text(7.55, -24.3, r'(b)', color='black', size=30)

ax1.legend(handles[midpoint:], labels[midpoint:], fontsize=21, loc='upper right', ncol=3, frameon=True)

# Histogram of the M_1450 values
bins = np.arange(-29.25, -23.5, 0.2) # the bin width is 0.2
ax3.hist(M1450_plot, bins=bins, edgecolor='black', alpha=0.8, color='darkgoldenrod', orientation='horizontal')
ax3.axhline(np.median(M1450_plot), color='darkred', linestyle='--', linewidth=3, label=r'M$_{1450,\rm{median}}$')
ax3.set_xticks([0, 3, 6])
ax3.text(4., -28.7, r'(d)', color='black', size=30)
ax3.get_yaxis().set_visible(False)
ax3.tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=28)
ax3.set_xlabel(r'$\rm{N_{QSOs}}$', size=30)

plt.tight_layout()
# plt.savefig('../../figures/J-Mag-z_plot_hist.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()
