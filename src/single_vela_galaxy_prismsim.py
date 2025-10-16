import numpy as np
# from scipy.interpolate import interp1d
# import synphot as syn
# from astropy import units as u
# from astropy import coordinates
from astropy.io import fits
# from pysiaf.utils import rotations
import matplotlib.pyplot as plt
# import joblib
# import concurrent.futures

# from ilia.tools import drizzle as drz
# from ilia.tools.time import convertTime
# from ilia.tools import astro
# from ilia import constants
# from ilia.environment.background import Background
# from ilia.environment.scene import Scene
# from ilia.instrument.disperser import RomanGrism, RomanPrism, RomanFilter
# from ilia.instrument.throughput import RomanEffectiveArea
# from ilia.instrument.psf import getPSFFoV, StandardRomanPSF

# import re
import sys
import os
# import time
import glob

vela_datadir = '/Volumes/Joshi_external_HDD/vela-datacubes/'

imglist = glob.glob(vela_datadir + '*_fnu.fits')
parlist = glob.glob(vela_datadir + '*_par.fits')

# Plot the distribution of redshifts in our simulations
plotzdist = False
if plotzdist:
    redshifts = []
    for file in imglist:
        himg = fits.open(file)
        redshifts.append(float(himg[0].header['ZRED']))
        himg.close()

    print('Min and max redshifts in the VELA sims:',
          np.min(redshifts), np.max(redshifts))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Number')
    ax.hist(np.asarray(redshifts), 30, range=(0.5, 2.0),
            histtype='step')
    fig.savefig('velasims_redshift_dist.png')

# ------------
# BEGIN SIMULATION BASED ON VELA DATACUBES

# First thing we need to do is decide a roll angle
rollangle_arr = np.array([0.0, 60.0, 120.0, 210.0, 300.0])
rollangle = np.random.choice(rollangle_arr)

# decide total number of galaxies to put in
totalgalaxies = np.random.randint(low=150, high=200)
print('Simulating total', totalgalaxies, 'galaxies at a roll angle of',
      rollangle, 'degrees.')

# coordinate limits within which to place galaxies within an SCA
# in this case we're using the central 1Kx1K area
scalimit_lo = 2048.0 - 500.0
scalimit_hi = 2048.0 + 500.0
# We're using the same array to choose x and y loc
# because the lo and hi limits are the same
# loc_choice = np.arange(scalimit_lo, scalimit_hi+1.0, 0.1)
loc_choice = np.arange(250, 750)
# array from which we will randomly choose SN loc
# these are coords relative to the host location
snloc_choice = np.arange(-50.0, 50.0)

# Get the number of wavelength elements in the datacubes
# that we care about. This should be the same for each
# datacube.
examplecube = fits.open(imglist[0])
cubewav = examplecube[2].data
wavidx = np.where((cubewav >= 6000) & (cubewav <= 20000))[0]
cubewav = cubewav[wavidx]
numwav = len(wavidx)
print('Wavelengths in cube:', numwav)
print('Datacube wavelength array:', cubewav)

# First let's make an oversampled "big" datacube
# by putting together all the datacubes
# Need an empty array first
sca_osamp_arr = np.zeros((numwav, 1000, 1000))

for i in range(20):#totalgalaxies):
    # First pick the x,y location at which this galaxy goes
    xloc = np.random.choice(loc_choice)
    yloc = np.random.choice(loc_choice)

    # Now pick a random datacube from the image list
    chosen_datacube = np.random.choice(imglist)

    # Now assign a SN location and brightness
    # SN location must be randomly chosen within some distance of host
    # SN brightness chosen randomly within some relative range of host mag
    snloc_x = xloc + np.random.choice(snloc_choice)
    snloc_y = yloc + np.random.choice(snloc_choice)

    # compute radial distance from host
    radialdist = np.sqrt((xloc - snloc_x)**2 + (yloc - snloc_y)**2)

    print('\nPlacing galaxy', os.path.basename(chosen_datacube),
          'datacube center at x,y:',
          '{:.2f}'.format(xloc), '{:.2f}'.format(yloc), '\n',
          'SN location at:',
          '{:.2f}'.format(snloc_x), '{:.2f}'.format(snloc_y), '; ',
          '{:.2f}'.format(radialdist), 'pixels radially from host.')

    # Put in the SN SED



    # Put the chosen datacube in the array
    # Read in datacube
    chdu = fits.open(chosen_datacube)
    datacube = chdu[1].data
    datacube = datacube[wavidx, :, :]
    halfsize = int(datacube.shape[2] / 2)
    sca_osamp_arr[:, int(yloc)-halfsize:int(yloc)+halfsize,
                  int(xloc)-halfsize:int(xloc)+halfsize] += datacube

    # close datacube
    chdu.close()

# Save the oversampled big datacube with all galaxies in it
phdu = fits.PrimaryHDU(data=sca_osamp_arr)
phdu.writeto('sca_osamp_bigcube.fits', overwrite=True)

# =============================
# Now call Ilia



sys.exit(0)
