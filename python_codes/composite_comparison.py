# Compare the composite spectra obtained in different redshift bins, magnitude bins, with and without BALs

import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import gridspec
from IPython import embed

# ---------------------------------------------------------------------------------
bal = True # True if you want to plot the composite spectrum with and without BALs
z_cut = False # True if you want to plot the 2 composite spectra obtained in the 2 redshift bins
mag_cut = False # True if you want to plot the 2 composite spectra obtained in the 2 magnitude bins

if bal:
    # This part is to plot the 2 composite spectra generated and saved (with and without BALs)
    hdu = fits.open('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/composite/composite_Onorato_dv110_bal.fits', fix=True, ignore_missing_simple=True)
    hdu1 = fits.open('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/composite/composite_Onorato_dv110.fits', fix=True, ignore_missing_simple=True)

elif z_cut:
    hdu = fits.open('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/composite/composite_Onorato_dv110_lowz.fits', fix=True, ignore_missing_simple=True)
    hdu1 = fits.open('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/composite/composite_Onorato_dv110_highz.fits', fix=True, ignore_missing_simple=True)

elif mag_cut:
    hdu = fits.open('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/composite/composite_Onorato_dv110_faint.fits', fix=True, ignore_missing_simple=True)
    hdu1 = fits.open('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/composite/composite_Onorato_dv110_bright.fits', fix=True, ignore_missing_simple=True)

spec = hdu[1].data
wave, flux, err, nused, zmean = spec['Wavelengths'], spec['Flux'], spec['Error'], spec['Nspec'], spec['Mean_z']

spec1 = hdu1[1].data
wave1, flux1, err1, nused1, zmean1 = spec1['Wavelengths'], spec1['Flux'], spec1['Error'], spec1['Nspec'], spec1['Mean_z']

wave_nomask = 1225.0
# Write some relevant emission lines wavelengths
line_names = [r'Ly$\alpha$', 'NV', 'SiII', 'OI/SiII', 'SiIV/OIV]', 'CIV', 'CIII]', 'MgII']
line_waves = [1215.67, 1240.81, 1260.4221, 1302.168, 1393.76, 1548.19, 1908.734, 2796.35]

fx = plt.figure(figsize=(17, 9))
gs = gridspec.GridSpec(12, 5, figure=fx, wspace=0., hspace=0.)
ax = fx.add_subplot(gs[4:12, 0:5])
ax1 = fx.add_subplot(gs[2:4, 0:5], sharex=ax)
ax2 = fx.add_subplot(gs[0:2, 0:5], sharex=ax)

ax.axvspan(min(wave), wave_nomask, alpha=0.3, color='dimgrey', label=r'$\lambda_{NOmasks}<$' + r'{:.0f}'.format(wave_nomask) + ' ' + r'$\rm{\AA}$')
ax1.axvspan(min(wave), wave_nomask, alpha=0.3, color='dimgrey', label='mask-free region')

ax1.plot(wave1, nused1, drawstyle='steps-mid', color='navy', alpha=0.8, zorder=12)
ax1.plot(wave, nused, drawstyle='steps-mid', color='darkorange', alpha=0.8, zorder=11)

ax2.plot(wave1, zmean1, drawstyle='steps-mid', color='navy', alpha=0.8, zorder=12)
ax2.plot(wave, zmean, drawstyle='steps-mid', color='darkorange', alpha=0.8, zorder=11)

if bal:
    ax.plot(wave1, flux1, drawstyle='steps-mid', color='navy', label='Excluding BALs', zorder=12, alpha=0.8)
    ax.plot(wave, flux, drawstyle='steps-mid', color='darkorange', label='Including BALs', alpha=0.8, zorder=11)
elif z_cut:
    ax.plot(wave1, flux1, drawstyle='steps-mid', color='navy', label=r'From $z \geq 6.70$ QSOs', zorder=12, alpha=0.8)
    ax.plot(wave, flux, drawstyle='steps-mid', color='darkorange', label=r'From $z<6.70$ QSOs', alpha=0.8, zorder=11)
elif mag_cut:
    ax.plot(wave1, flux1, drawstyle='steps-mid', color='navy', label=r'From $M_{1450} < -26.0$ QSOs', zorder=12, alpha=0.8)
    ax.plot(wave, flux, drawstyle='steps-mid', color='darkorange', label=r'From $M_{1450} \geq -26.0$ QSOs', alpha=0.8, zorder=11)

ax.plot(wave1, err1, drawstyle='steps-mid', color='royalblue', zorder=12, alpha=0.7)
ax.plot(wave, err, drawstyle='steps-mid', color='lightcoral', zorder=11, alpha=0.7)


for pos, name in zip(line_waves, line_names):
    ax.axvline(x=pos, color='black', linestyle='--', linewidth=0.75)
    ax.annotate(name, xy=(pos, 4.5), xytext=(-10.5, 20), textcoords='offset points', rotation=90, fontsize=10, color='black')

# Inset panel with a zoom
axins = ax.inset_axes([0.4, 0.25, 0.35, 0.74])
axins.plot(wave, flux, drawstyle='steps-mid', color='darkorange', label='Including BAL', alpha=0.8, zorder=11)
axins.plot(wave, err, drawstyle='steps-mid', color='lightcoral', zorder=11, alpha=0.7)
axins.plot(wave1, flux1, drawstyle='steps-mid', color='navy', label='Excluding BAL', zorder=12, alpha=0.8)
axins.plot(wave1, err1, drawstyle='steps-mid', color='royalblue', zorder=12, alpha=0.7)
axins.set_xlim(1175., 1580.)
axins.set_ylim(-0.05, 5.01)
axins.set_xticks([1200.,1300.,1400.,1500.])
axins.set_yticks([0.,1.,2.,3.,4.])
axins.set_facecolor('whitesmoke')
axins.tick_params(direction='in', length=6, width=1, which='both', right=True, top=True, labelsize=17)


ax.axis([1038.,3338.,-0.05,5.4])
ax.set_xlabel(r'Rest-frame Wavelength [$\rm{\AA}$]', size=23)
ax.set_ylabel(r'Flux', size=23)
ax.legend(fontsize=15, loc='upper right')
ax.tick_params(direction='in', length=6, width=1, which='both', right=True, top=False, labelsize=20)

ax1.set_ylabel(r'$\rm{N_{spectra}}$', size=23)
if bal:
    ax1.set_yticks([0, 20, 45])
    ax1.axis([1038.,3338.,-0.05,52.])
elif z_cut or mag_cut:
    ax1.set_yticks([0, 8, 17])
    ax1.axis([1038.,3338.,-0.05,22.])
ax1.tick_params(direction='in', length=6, width=1, which='both', right=True, top=False, labelsize=20)

ax2.set_ylabel(r'$\rm{z_{mean}}$', size=23)
if bal:
    ax2.set_yticks([6.6,6.9,7.2])
    ax2.axis([1038.,3338.,6.49,7.3])
elif z_cut or mag_cut:
    ax2.set_yticks([6.6,7.0,7.4])
    ax2.axis([1038.,3338.,6.49,7.56])
ax2.tick_params(direction='in', length=6, width=1, which='both', right=True, top=False, labelsize=20)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.tight_layout()
# plt.savefig('../../figures/composite_comparison.png', bbox_inches='tight', dpi=500) #Change the name of the file
plt.show()