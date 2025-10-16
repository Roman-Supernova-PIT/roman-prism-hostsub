import numpy as np
from astropy.io import fits

import param_fit as pf
import sys
import os

home = os.getenv('HOME')


if __name__ == '__main__':

    # User defined params
    hostsubdir = home + '/Documents/Roman/PIT/prism/hostlight_subtraction/'
    datadir = hostsubdir + 'simdata_prism_galsn/'
    prismsciextnum = 1
    dirsciextnum = 1
    prismimgname = 'test_2d_multipleGalaxies_WFI_WFI01_FULL_prism_rollAngle120.fits'
    dirimgname = 'test_2d_multipleGalaxies_WFI_WFI01_FULL_F106_rollAngle120.fits'
    # ===============

    # Read in prism and direct images
    phdu = fits.open(datadir + prismimgname)
    prismdata = phdu[prismsciextnum].data

    dhdu = fits.open(datadir + dirimgname)
    dirimgdata = dhdu[dirsciextnum].data

    # 

    sys.exit(0)
