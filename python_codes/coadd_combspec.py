# Code to coadd the spectra from the different instruments in case we have an echelle and a long slit spectrum.

import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
from astropy.table import Table
from pypeit.core import coadd
from pypeit import utils
from pypeit.core.wavecal import wvutils
from pypeit.onespec import OneSpec
from pypeit.history import History
from astropy import constants as const
from highz_qso_redux.utils.utils import open_spectra, normalize_spectra

c_kms = const.c.to('km/s').value

# qso_name = 'J0411-0907'
# qso_name = 'J0218+0007'
# qso_name = 'J0706+2921'
#qso_name = 'J0319-1008'
#qso_name = 'J0038-0653'
qso_name = 'J1917+5003'
# qso_name = 'J1058+2930'

# instrum = 'XShooter_NIR'
instrum = 'NIRES'

if qso_name == 'J0411-0907' or qso_name == 'J1917+5003' or qso_name == 'J1058+2930':
    instrum1 = 'MODSR'
elif qso_name == 'J0218+0007':
    instrum1 = 'LRISR'
elif qso_name == 'J0706+2921':
    instrum1 = 'DEIMOS'
elif qso_name == 'J0319-1008':
    instrum1 = 'GMOS'
elif qso_name == 'J0038-0653':
    instrum1 = 'XShooter_VIS'

ground_path = '/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/PypeIt_data/REDUX_OUT/spectra_for_plotting_coadd'
ir_filename = '%s_%s_coadd_tellcorr.fits' %(qso_name, instrum)
ir_file = os.path.join(ground_path, ir_filename)
ir_table = Table.read(ir_file)
optical_filename = '%s_%s_coadd_tellcorr.fits' %(qso_name, instrum1)
optical_file = os.path.join(ground_path, optical_filename)
optical_table = Table.read(optical_file)
outfile = os.path.join(ground_path, '%s_%s-%s_coadd_tellcorr.fits' %(qso_name, instrum, instrum1))



plt.plot(ir_table['wave_grid_mid'], ir_table['flux']*ir_table['mask'],drawstyle='steps-mid', color='black', label='IR')
plt.plot(ir_table['wave_grid_mid'], np.sqrt(utils.inverse(ir_table['ivar']))*ir_table['mask'],drawstyle='steps-mid', color='cyan', label='IR')
plt.plot(optical_table['wave_grid_mid'], optical_table['flux']*optical_table['mask'],drawstyle='steps-mid', color='orange', label='OPT')
plt.plot(optical_table['wave_grid_mid'], np.sqrt(utils.inverse(optical_table['ivar']))*optical_table['mask'],drawstyle='steps-mid', color='magenta', label='OPT')
# plt.plot(optical_table['OPT_WAVE'], optical_table['OPT_FLAM']*optical_table['OPT_MASK'],drawstyle='steps-mid', color='orange')
# plt.plot(optical_table['OPT_WAVE'], optical_table['OPT_FLAM_SIG']*optical_table['OPT_MASK'],drawstyle='steps-mid', color='magenta')
plt.show()

norm_wave = 9800.#10620.#10172.
flux_norm_ir, ivar_norm_ir = normalize_spectra(ir_table['wave_grid_mid'], ir_table['flux'], ir_table['ivar'], norm_wave)
flux_norm_opt, ivar_norm_opt = normalize_spectra(optical_table['wave_grid_mid'], optical_table['flux'], optical_table['ivar'], norm_wave)

# Mask the optical above wave_max_opt where it is pure noise and the IR below wave_max_ir where it is pure noise.
wave_max_opt=9900.#10200.#10172.
wave_max_ir=9900.
opt_gpm = optical_table['wave_grid_mid'] < wave_max_opt
ir_gpm = ir_table['wave_grid_mid'] > wave_max_ir
waves  = [optical_table['wave_grid_mid'][opt_gpm], ir_table['wave_grid_mid'][ir_gpm]]
fluxes = [flux_norm_opt[opt_gpm], flux_norm_ir[ir_gpm]]
ivars =  [ivar_norm_opt[opt_gpm], ivar_norm_ir[ir_gpm]]
gpms =   [optical_table['mask'][opt_gpm].astype(bool), ir_table['mask'][ir_gpm].astype(bool)]


dwave, dloglam, resln_guess, pix_per_sigma = [], [], [], []
for wave in waves:
    _dwave, _dloglam, _resln_guess, _pix_per_sigma = wvutils.get_sampling(wave)
    dwave.append(_dwave)
    dloglam.append(_dloglam)
    resln_guess.append(_resln_guess)
    pix_per_sigma.append(_pix_per_sigma)

dv_samp = [dlog10lam*c_kms*np.log(10.0) for dlog10lam in dloglam]

# NIRES has 38 km/s pixels.
# MODS has 33 km/s pixels.
# LRIS has 56 km/s pixels.
# DEIMOS has 17 km/s pixels.
# GMOS has 51 km/s pixels.
# X-Shooter has 11-13 km/s pixels.

if instrum1 == 'MODSR' or instrum1 == 'DEIMOS':
    dv = 40.0
elif instrum1 == 'LRISR':
    dv = 60.0
elif instrum1 == 'GMOS':
    dv = 55.0
elif instrum1 == 'XShooter_VIS':
    dv = 13.0

wave_grid_min = 8000.0
wave_grid_max = 24700.0
debug=False
show=True

wave_grid_mid, wave_coadd, flux_coadd, ivar_coadd, gpm_coadd = coadd.multi_combspec(
    waves, fluxes, ivars, gpms, sn_smooth_npix=None, scale_method='poly', wave_method='log10', dv=dv,
    wave_grid_min=wave_grid_min, wave_grid_max=wave_grid_max, weight_method='wave_dependent', lower=3.0, upper=3.0,
    debug=debug, debug_scale=debug, show_scale=debug, show=show)

# Generate a bogus telluric by stitching the two together
telluric_ir = np.ones_like(wave_grid_mid)
itell_ir = (wave_grid_mid >= ir_table['wave_grid_mid'].min()) & (wave_grid_mid <= ir_table['wave_grid_mid'].max())
telluric_ir[itell_ir] = np.interp(wave_grid_mid[itell_ir], ir_table['wave_grid_mid'], ir_table['telluric'])

telluric_opt = np.ones_like(wave_grid_mid)
itell_opt = (wave_grid_mid >= optical_table['wave_grid_mid'].min()) & (wave_grid_mid <= optical_table['wave_grid_mid'].max())
telluric_opt[itell_opt] = np.interp(wave_grid_mid[itell_opt], optical_table['wave_grid_mid'], optical_table['telluric'])


# Now use a sigmoid to create a hybrid spectrum that is a linear combination of the telluric_ir and telluric_opt
sigmoid_wave_cut = 9500.0
sigmoid_wave_width = 100.0
wave_sigmoid = scipy.special.expit((wave_grid_mid - sigmoid_wave_cut)/sigmoid_wave_width)
telluric_tot = telluric_opt*(1.0 -wave_sigmoid) + telluric_ir*wave_sigmoid

onespec = OneSpec(wave=wave_coadd, wave_grid_mid=wave_grid_mid, flux=flux_coadd,
                  telluric=telluric_tot,  PYP_SPEC='%s+%s' %(instrum1, instrum), ivar=ivar_coadd, sigma=np.sqrt(utils.inverse(ivar_coadd)),
                  mask=gpm_coadd.astype(int), ext_mode='OPT', fluxed=True)

# Add history entries for coadding.
history = History()
history.add_coadd1d([optical_file, ir_file],['%s' %instrum1, '%s' %instrum])

# Write
onespec.to_file(outfile, history=history, overwrite=True)