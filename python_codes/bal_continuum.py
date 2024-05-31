# Code to visualize what the continuum would be in a quasar with BAL features.

import numpy as np
from matplotlib import pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from highz_qso_redux.utils import utils
from highz_qso_redux.utils.utils import open_spectra, wave_0, flux_0, normalize_spectra, wave_obs
from qso_fitting.data.fluxing.flux_correct import spec_interp_gpm
from pypeit.utils import inverse
from highz_qso_redux.scripts.flux_cal.Jinyi_flux_cal_functions import m1450

path = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/PypeIt_data/REDUX_OUT/spectra_for_plotting_coadd/new_spec'
# --------------------------------------------------------------
qso_name = 'J0910-0414' #BAL 1
# qso_name = 'J0923+0402' #BAL 2

if qso_name == 'J0910-0414':
    instrum = 'NIRES-GNIRS'
    z = 6.6363
elif qso_name == 'J0923+0402':
    instrum = 'XShooter_VIS-NIR'
    z = 6.633

spec_name = '%s_%s_coadd_tellcorr.fits' % (qso_name, instrum)
wave, flux, ivar, gpm, telluric = open_spectra(path, spec_name, telluric=True, flux_scale=True)
wave_rest = wave_0(wave, z)

flux_interp, ivar_interp = spec_interp_gpm(wave_rest, flux, gpm, sigma_or_ivar=ivar)

# Load the composite spectrum to make the correction
hdu1 = fits.open('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/composite/composite_Onorato_dv110.fits', fix=True, ignore_missing_simple=True)
spec1 = hdu1[1].data
wave1, flux1, err1, nused1, zmean1 = spec1['Wavelengths'], spec1['Flux'], spec1['Error'], spec1['Nspec'], spec1['Mean_z']

wave_norm_rest = 2000.0

if qso_name == 'J0910-0414':
    exp = 0.3
elif qso_name == 'J0923+0402':
    exp = 0.8
# 0.8 for J0923+0402, 0.3 for J0910-0414 work well

factor = flux_interp[np.argmin(np.abs(wave_rest - wave_norm_rest))] / flux1[np.argmin(np.abs(wave1 - wave_norm_rest))]
new_flux_comp = flux1 * factor * (wave1/wave_norm_rest)**(exp)

# Move the composite to the observed frame
wave1_obs = wave_obs(wave1, z)

m_1450_spec, M_1450_spec = m1450(wave1_obs, new_flux_comp, z, cosmo)
# print('M1450 if lambda_norm=%.1f and exp=%.2f: '%(wave_norm_rest, exp), M_1450_spec)

# Smooth for visualization
window = 10
flux_sm, ivar_sm = utils.ivarsmooth(flux_interp, ivar_interp, window=window)
sigma = np.sqrt(inverse(ivar_sm))

fig = plt.figure(figsize=(16, 7))
ax = fig.add_subplot(111)

#ax.plot(wave_rest, flux_sm, drawstyle='steps-mid', color='royalblue', zorder=9, label=qso_name, alpha=0.8)
if qso_name == 'J0910-0414':
    ax.plot(wave_rest, flux_sm, drawstyle='steps-mid', color='royalblue', zorder=9, label=r'J0910$-$0414', alpha=0.8)
elif qso_name == 'J0923+0402':
    ax.plot(wave_rest, flux_sm, drawstyle='steps-mid', color='royalblue', zorder=9, label=r'J0923$+$0402', alpha=0.8)
ax.plot(wave1, new_flux_comp, drawstyle='steps-mid', color='darkorange', zorder=10, label='Composite', alpha=0.8)
ax.plot(wave_rest, sigma, color='midnightblue', zorder=8, alpha=0.6)
ax.plot(wave1, err1, color='peru', zorder=7, alpha=0.6)

# Plot a vertical line at lambda 1450 A
ax.axvline(x=1450.0, color='black', linestyle='--', alpha=1, label=r'$\lambda_{1450}$')
# Plot a vertical line at lambda norm
ax.axvline(x=wave_norm_rest, color='limegreen', linestyle='--', alpha=1, label=r'$\lambda_{norm}=%.f \rm{\AA}$'%wave_norm_rest)

if qso_name == 'J0910-0414':
    ax.scatter(1450.0, 0.55, marker='*', color='crimson', s=250, zorder=15)
elif qso_name == 'J0923+0402':
    ax.scatter(1450.0, 0.62, marker='*', color='crimson', s=250, zorder=15)

ax.set_xlabel(r'Rest-Frame Wavelength [$\rm{\AA}$]', size=22)
ax.set_ylabel(r'Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\rm{\AA}^{-1}$]', size=22)
# ax.axis([8430., 25190., -0.03, 1.1])
ax.axis([1080., 3249., -0.03, 2.7])
ax.legend(loc='upper right', fontsize=20)
ax.tick_params(direction='in', length=6, width=1, which='both', right=True, top=False, labelsize=18)
plt.tight_layout()
# plt.savefig('../../figures/%s_bal-composite_%.i-%.2f.png'%(qso_name[0:5],wave_norm_rest, exp), dpi=300)
plt.show()