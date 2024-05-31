# Auto-generated Flux input file using PypeIt version: 1.12.3.dev235+g648f92808
# UTC 2023-10-12T11:57:15.361

# User-defined execution parameters
[fluxcalib]
  extinct_correct = False  # Set to True if your SENSFUNC derived with the UVIS algorithm
# Please add your SENSFUNC file name below before running pypeit_flux_calib

# Data block 
flux read
 path Science_coadd
                                              filename | sensfile
spec1d_N20180625S0286-N20180729S0030-HSC1243+0100.fits | ../sens_N20200102S0074-GD71_GNIRS_20191231T224213.756.fits        
flux end

