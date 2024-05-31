# Script to create the file to query the KOA archive

import os
from astropy.table import Table
import astropy.units as units
from astropy.coordinates import SkyCoord

qso_table_file = os.path.join('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/master_list/', 'qso_rsg6_v2.0_all_v20220401.fits')
qso_table_all = Table.read(qso_table_file)

z_cut1 = 6.50
#z_cut2 = 7.70

indx = (qso_table_all['rs_best']>= z_cut1) & (qso_table_all['rs_best'] < max(qso_table_all['rs_best']) )
qso_table_cut = qso_table_all[indx]

# Write the Keck File
with open('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/sample/keck_query_python1.txt', 'w') as f:
    for row in qso_table_cut:
        oldcord = SkyCoord(row['RA'], row['DEC'], frame='icrs', unit=(units.hourangle, units.deg))
        ra_new, dec_new = oldcord.ra.deg, oldcord.dec.deg
        line = '{:s}\t{:.9f}\t{:.9f}'.format(row['JName'], ra_new, dec_new)
        f.write(line)
        f.write('\n')