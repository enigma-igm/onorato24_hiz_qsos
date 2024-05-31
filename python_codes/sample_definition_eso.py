# Script to create the file to query the ESO archive

from IPython import embed
import os
import numpy as np
from astropy.table import Table

qso_table_file = os.path.join('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/master_list/', 'qso_rsg6_v2.0_all_v20220401.fits')
qso_table_all = Table.read(qso_table_file)

z_cut1 = 6.49
z_cut2 = 7.65
idx = np.ones(len(qso_table_all), dtype=bool)

excluded_terms = ['J003803.782m065322.87', 'J022935.246m080822.987', 'J024401.020-500853.700', 'J173243.12p653113.50', 'J052559.675m240622.980', 'J014852.694m282639.329', 'J222228.15m154050.84', 'J005537.064m342636.267', 'J010650p060725', 'J225634.63m024559.18']

for i, qso_table in enumerate(qso_table_all):
    if 'atsuoka' in qso_table['Ref']:
        idx[i] = False
        if 'J124353.930p010038.500' in qso_table['JName']:
            idx[i] = True
        if 'Wang' in qso_table['Ref']:
            idx[i] = True
    elif any(term in qso_table['JName'] for term in excluded_terms):
        idx[i] = False

qso_ref_cut = qso_table_all[np.where(idx == True)]
qso_table_cut = qso_ref_cut[(qso_ref_cut['rs_best']> z_cut1) & (qso_ref_cut['rs_best']<= z_cut2)]

# Write the ESO File
with open('/Users/silviaonorato/Projects/highz_qso_redux/highz_qso_redux/tables/sample/eso_query_python.txt', 'w') as f:
    for row in qso_table_cut:
        line = '{:s}\t{:s}'.format(row['RA'].replace(':',' '), row['DEC'].replace(':',' ')) + ' # z = {:5.3f}; ref = {:s}'.format(row['rs_best'], row['Ref'])
        f.write(line)
        f.write('\n')