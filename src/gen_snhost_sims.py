import numpy as np
from scipy.interpolate import interp1d
import synphot as syn
from astropy import units as u
from astropy import coordinates
from astropy.io import fits
from pysiaf.utils import rotations
# import matplotlib.pyplot as plt
# from ilia.tools import plotting
import joblib
import concurrent.futures

from ilia.tools import drizzle as drz
from ilia.tools.time import convertTime
from ilia.tools import astro
from ilia import constants
from ilia.environment.background import Background
from ilia.environment.scene import Scene
from ilia.instrument.disperser import RomanGrism, RomanPrism, RomanFilter
from ilia.instrument.throughput import RomanEffectiveArea
from ilia.instrument.psf import getPSFFoV, StandardRomanPSF

import re
import sys
import os
import time

home = os.getenv('HOME')
hostsubdir = home + "/Documents/Roman/PIT/prism/hostlight_subtraction/"
datacubeFilename = hostsubdir + "fromMassimo/gal_cube_os3_3D.fits"

DEF_COORDS = coordinates.SkyCoord(7.60244299, -44.79116827,
                                  frame='icrs', unit='deg')
ROMAN_PIXSCALE = constants.ROMAN_PIXSCALE
SN_x = 95
SN_y = 70
scale_factor = 20  # scaling factor for SN flux
exposureTime = 900.0 * u.s

subtractBackground = True
nCPUs = 6

print('\n')
print('Roman pixel scale assumed:', ROMAN_PIXSCALE)
# ==============
# Open datacube
hduList = fits.open(datacubeFilename)
datacube = hduList[0].data[:, 0:-1, 0:-1] * syn.units.FNU

osamp = int(hduList[0].header['OVERSAMP'])
lambdaMin = float(hduList[0].header['LBDAMIN'])
lambdaMax = float(hduList[0].header['LBDAMAX'])
lambdaStep = float(hduList[0].header['LBDASTEP'])
redshift = hduList[0].header['REDSHIFT']

wavelength = (np.arange(lambdaMin, lambdaMax+0.1*lambdaStep, lambdaStep)
              * u.angstrom)

# Effective pixel scale for the oversampled datacube
eff_pixscl = ROMAN_PIXSCALE / float(osamp)

# load SN SED
SN = np.loadtxt(hostsubdir + 'simdata_prism_galsn/lfnana_fnu.txt')
SN_SED = interp1d(SN[:, 0], SN[:, 1], kind='linear')

datacube[:, SN_y - 1, SN_x - 1] += (scale_factor
                                    * SN_SED(wavelength) * syn.units.FNU)
print('Inserted SN SED at datacube coords (DS9 x, y):', SN_x, SN_y)
print('Inserted SN SED at datacube coords (row, col):', SN_y - 1, SN_x - 1)

# Get coordinates where the center (?) of the modified datacube
# with the SN SED inserted will be placed in the Roman exposure
c1 = DEF_COORDS.spherical_offsets_by(0.0 * u.arcsec,
                                     0.0 * u.pix * ROMAN_PIXSCALE)
# c2 = DEF_COORDS.spherical_offsets_by(50.0 * u.pix * ROMAN_PIXSCALE,
#                                      0.0 * u.arcsec)
# c3 = DEF_COORDS.spherical_offsets_by(-25.0 * u.pix * ROMAN_PIXSCALE,
#                                      -50.0 * u.pix * ROMAN_PIXSCALE)

coordinatesList = [c1]

print('\nCoordinates where the center (?) of the modified datacube',
      'with the SN SED inserted will be placed in the Roman exposure:')
print(coordinatesList)

# use the two lines below for an array of roll angles
delAngle = 10.0
rollAngles = coordinates.Angle(np.arange(0.0, 10.0, delAngle),
                               unit='deg').wrap_at(360.0 * u.deg)

# use this line for just the 0 deg roll angle
# rollAngles = coordinates.Angle([0.0], unit='deg').wrap_at(360.0 * u.deg)

print('Roll angles:', rollAngles)

# Set up aperture list and reference coordinates
(rsiaf, roman_apertures,
 V2Ref_roman, V3Ref_roman, pa_y_v3) = astro.getRomanApertures()

alpha = np.array([c.ra.deg for c in coordinatesList])
delta = np.array([c.dec.deg for c in coordinatesList])

# Find the central coordinates of SCA01 in the v2-v3 plane
wfi01_v2, wfi01_v3 = roman_apertures[0].idl_to_tel(0, 0)
print('SCA 01 center coords (v2, v3):', wfi01_v2, wfi01_v3)
print('\n')

"""
# Now we plot the detector orientations as a function of roll angle,
# i.e. the position angle between -x axis and the positive RA axis,
# measured positive towards NORTH
nCols = 6
nRows = int(np.ceil(rollAngles.size / nCols))

xSize = 13
ySize = nRows * xSize / nCols

print(nRows, nCols, xSize, ySize)

showLabel = False

factor = 800.0

# Now we rotate the detector around this point and plot
fig, axes = plt.subplots(figsize=(xSize, ySize),
                         ncols=nCols, nrows=nRows,
                         sharex=True, sharey=True)

for i in range(nRows):
    for j in range(nCols):
        k = np.ravel_multi_index((i, j), (nRows, nCols))
        if (k < rollAngles.size):
            rollAngle = rollAngles[k]
            pa_v3 = (360.0 * u.deg - pa_y_v3 - rollAngle).wrap_at(360.0*u.deg)
            pa_aper = (pa_y_v3 + pa_v3).wrap_at(360.0 * u.deg)

            att = rotations.attitude(wfi01_v2, wfi01_v3, DEF_COORDS.ra,
                                     DEF_COORDS.dec, pa_v3)
            for aperture in roman_apertures:
                aperture.set_attitude_matrix(att)
                pointing_idl, in_footprint_pointing = \
                    astro.footprintContains(aperture,
                                            DEF_COORDS.ra, DEF_COORDS.dec,
                                            'idl')
                coords_idl, in_footprint_coords = \
                    astro.footprintContains(aperture,
                                            alpha, delta, 'idl')
                if (in_footprint_pointing.any() and in_footprint_coords.any()):
                    print(rollAngle.value, pa_v3.value, pa_aper.value,
                          aperture.AperName, in_footprint_pointing,
                          in_footprint_coords)
                    axes[i, j].plot(pointing_idl[:, 0],
                                    pointing_idl[:, 1], 'rx')
                    axes[i, j].plot(coords_idl[:, 0], coords_idl[:, 1], 'k.')
            axes[i, j].text(0.05, 0.95,
                            r'${0:0.0f}^\circ$'.format(rollAngle.value),
                            ha='left', va='top',
                            transform=axes[i, j].transAxes)
        else:
            axes[i, j].set_visible(False)
axCommon = plotting.drawCommonLabel(r'$X$ [arcsec]',
                                    r'$Y$ [arcsec]', fig,
                                    xPad=20, yPad=45)

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig('test_rollAngle_galaxyTriplets.png', bbox_inches='tight', dpi=300)
print("Plotting all done!")
"""

scene = Scene(datacube, wavelength,
              pixScale=ROMAN_PIXSCALE / osamp,
              oversample=osamp)

nP, nQ = scene.nP, scene.nQ

nWX = 2 * nQ - 1
nWY = 2 * nP - 1

FoV = getPSFFoV(nP, nQ, osamp)

print("Field of View:", FoV)

PSF = StandardRomanPSF(FoV, osamp)

AEff = RomanEffectiveArea(pandeia=False, useNew=True)

filterNames = ['F106', 'prism']

bg = Background()

# for k in range(0, 2):
startTimeAll = time.time()
for k, rollAngle in enumerate(rollAngles):
    rollAngle = rollAngles[k]
    pa_v3 = (360.0 * u.deg - pa_y_v3 - rollAngle).wrap_at(360.0 * u.deg)
    pa_aper = (pa_y_v3 + pa_v3).wrap_at(360.0 * u.deg)

    for filterName in filterNames:
        weight = 1.0 / scene.oversample
        if (filterName == 'grism'):
            disperser = RomanGrism(oversample=osamp, centered=True)
        elif (filterName == 'prism'):
            disperser = RomanPrism(oversample=osamp, centered=True)
        else:
            disperser = RomanFilter()
            weight = 1.0
            if (rollAngle != 0):
                continue

        wavePix = disperser.getWavelengthPixels()
        pixCoords = disperser.getPixelCoordinates()
        dispersion = disperser.getDispersionCurve()

        psf = PSF.getResampledPSF(wavePix)
        effArea = AEff.getEffectiveArea(wavePix, filterName)
        nonZeroes = np.argwhere(effArea > 0).flatten()

        bkg = bg.getBackground(wavePix, oversample=osamp)
        bkg_flam = syn.units.convert_flux(wavePix, bkg, syn.units.PHOTLAM)
        bgPhotonRate = (bkg_flam * effArea * dispersion
                        / scene.oversample).decompose()

        bgRate = (bgPhotonRate.sum().value
                  * (scene.oversample ** 2) * u.adu / u.s)

        if (nCPUs is None) or (nCPUs <= 0):
            nSplit = joblib.cpu_count()
        else:
            nSplit = nCPUs

        nonZeroes_split = np.array_split(nonZeroes, nSplit)

        # print(nonZeroes_split)

        # import pdb; pdb.set_trace()

        scene.resampleScene(wavePix)

        # resampledScene = syn.units.convert_flux(wavePix,
        #                                         scene.getResampledScene(),
        #                                         syn.units.PHOTLAM)

        # randomShiftAlpha = (-1.0 + 2.0*np.random.rand()) * u.pix
        # randomShiftDelta = (-1.0 + 2.0*np.random.rand()) * u.pix
        dithers = [(0, 0)*u.pix]
        # ,(-50.21*u.pix, -25.52*u.pix),
        # (50.36*u.pix, -15.64*u.pix), (25.19*u.pix, 35.23*u.pix)]
        for ditherN in range(len(dithers)):

            randomShiftAlpha, randomShiftDelta = dithers[ditherN]
            pointing = DEF_COORDS.spherical_offsets_by(randomShiftAlpha
                                                       * ROMAN_PIXSCALE,
                                                       randomShiftDelta
                                                       * ROMAN_PIXSCALE)

            att = rotations.attitude(wfi01_v2, wfi01_v3,
                                     pointing.ra, pointing.dec, pa_v3)

            for aperture in roman_apertures:
                aperture.set_attitude_matrix(att)

                coords_idl, in_footprint_coords =\
                    astro.footprintContains(aperture, alpha, delta, 'idl')

                if in_footprint_coords.any():
                    print(k, pointing.ra, pointing.dec,
                          rollAngle.value, pa_v3.value,
                          pa_aper.value, aperture.AperName,
                          in_footprint_coords)

                    psfDrizzleMatrix = drz.getAreaFractionMatrix(psf.shape[2],
                                                                 psf.shape[1],
                                                                 0, 0, pa_aper,
                                                                 eff_pixscl,
                                                                 nXOut=nWX,
                                                                 nYOut=nWY,
                                                                 oversample=1)

                    naxis1, naxis2 = aperture.XSciSize, aperture.YSciSize
                    crpix1, crpix2 = aperture.XSciRef, aperture.YSciRef

                    # This is how we transform the crpix1, crpix2
                    # coordinate, which is a 1-indexed, full-integer
                    # image grid, into the coordinate of a zero-indexed,
                    # half-integer grid (the grid of pypolyclip)
                    xCOut, yCOut = crpix1 - 0.5, crpix2 - 0.5

                    downsample = 1.0 / float(osamp)

                    specRate = np.zeros((naxis2, naxis1))

                    def dropImage(wave_indices, dx, dy, angle):
                        if (filterName != 'grism') and (filterName != 'prism'):
                            drzMatrix =\
                                drz.getAreaFractionMatrix(nQ, nP, dx, dy,
                                                          angle,
                                                          eff_pixscl,
                                                          nXOut=naxis1,
                                                          nYOut=naxis2,
                                                          xCOut=xCOut,
                                                          yCOut=yCOut,
                                                          oversample=downsample)

                        thisSpecRate = np.zeros((naxis2, naxis1))
                        for l in wave_indices:
                            rotatedPSF = (psfDrizzleMatrix @ np.nan_to_num(psf[l]).flatten()).reshape((nWY, nWX))
                            convolvedImage =\
                                scene.getConvolvedResampledMonochromaticSlice(l, rotatedPSF)

                            photonRate = (convolvedImage * effArea[l] * dispersion[l] * weight).decompose()
                            photonRate = (photonRate * 1.0 * u.adu / u.ph).decompose()  # Conversion rate from photon to ADU

                            if (filterName == 'grism') or (filterName == 'prism'):
                                shiftY_arcsec = pixCoords[l] * ROMAN_PIXSCALE
                                drzMatrix =\
                                    drz.getAreaFractionMatrix(nQ, nP, dx,
                                                              dy + shiftY_arcsec,
                                                              angle, eff_pixscl,
                                                              nXOut=naxis1, nYOut=naxis2,
                                                              xCOut=xCOut, yCOut=yCOut, oversample=downsample)

                            thisSpecRate += np.nan_to_num((drzMatrix @ np.nan_to_num(photonRate).flatten()).reshape((naxis2, naxis1)))
                        return thisSpecRate

                    startTime = time.time()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=nCPUs) as executor:
                        jobList = []
                        for coords in coords_idl[in_footprint_coords]:
                            dx, dy = coords[0] * u.arcsec, coords[1] * u.arcsec

                            for thisNonZeroes in nonZeroes_split:
                                jobList.append(executor.submit(dropImage, thisNonZeroes, dx, dy, rollAngle))

                        print("Number of jobs:", len(jobList))

                        for future in concurrent.futures.as_completed(jobList):
                            specRate += future.result()

                    # Make values positive so that random poisson
                    # generator does not break
                    specRate = np.nan_to_num(np.clip(specRate, a_min=0, a_max=None))
                    specRate *= u.adu / u.s
                    noisySpec = np.random.poisson((specRate + bgRate).value * exposureTime.value) * u.adu

                    if subtractBackground:
                        noisySpec -= bgRate * exposureTime

                    stdDev = np.sqrt((specRate + bgRate) * exposureTime).value * u.adu

                    alpha0, delta0 = aperture.idl_to_sky(0, 0)
                    wcs = astro.getWCS(rollAngle, filterName, naxis2, naxis1,
                                       alpha=alpha0, delta=delta0,
                                       crpix1=crpix1, crpix2=crpix2,
                                       oversample=1)

                    hduListOut = []

                    header = fits.Header()
                    header['COMMENT'] = "Simulated Roman 2d spectra generated using Ilia"
                    header['COMMENT'] = "https://gitlab.com/astraatmadja/Ilia"

                    hduListOut.append(fits.PrimaryHDU(header=header))

                    image_hdu = fits.ImageHDU(noisySpec.value, name='SCI', ver=1)
                    image_hdu.header.set('TELESCOP', aperture.observatory)
                    image_hdu.header.set('INSTRUME', aperture.InstrName)
                    image_hdu.header.set('APERTURE', aperture.AperName)
                    image_hdu.header.set('SCA_NUM', int(re.sub("[^0-9]", "", aperture.AperName)))
                    image_hdu.header.set('FILTER', "{0:s}".format(filterName), 'filter')
                    image_hdu.header.set('BUNIT', str(noisySpec.unit), 'brightness units')
                    image_hdu.header.set('exptime', "{0:.3f}".format(exposureTime.value), 'exposure time in sec')
                    image_hdu.header.set('ORIENTAT', "{0:f}".format(pa_aper.deg))
                    image_hdu.header.set('PA_V3', "{0:f}".format(pa_v3.deg))
                    image_hdu.header.set('BG_RATE', "{0:f}".format(bgRate.value), 'sky background rate in ADU/s')
                    image_hdu.header.update(wcs.to_header())

                    hduListOut.append(image_hdu)

                    stdDevImage_hdu = fits.ImageHDU(stdDev.value, name='ERR', ver=1)
                    stdDevImage_hdu.header.set('TELESCOP', aperture.observatory)
                    stdDevImage_hdu.header.set('INSTRUME', aperture.InstrName)
                    stdDevImage_hdu.header.set('APERTURE', aperture.AperName)
                    stdDevImage_hdu.header.set('SCA_NUM', int(re.sub("[^0-9]", "", aperture.AperName)))
                    stdDevImage_hdu.header.set('FILTER', "{0:s}".format(filterName), 'filter')
                    stdDevImage_hdu.header.set('BUNIT', str(noisySpec.unit), 'brightness units')
                    stdDevImage_hdu.header.set('exptime', "{0:.3f}".format(exposureTime.value), 'exposure time in sec')
                    stdDevImage_hdu.header.set('ORIENTAT', "{0:f}".format(pa_aper.deg))
                    stdDevImage_hdu.header.set('PA_V3', "{0:f}".format(pa_v3.deg))
                    stdDevImage_hdu.header.set('BG_RATE', "{0:f}".format(bgRate.value), 'sky background rate in ADU/s')
                    stdDevImage_hdu.header.update(wcs.to_header())

                    hduListOut.append(stdDevImage_hdu)

                    DQImage_hdu = fits.ImageHDU(np.ones(noisySpec.shape, dtype=np.int16), name='DQ', ver=1)
                    DQImage_hdu.header.set('TELESCOP', aperture.observatory)
                    DQImage_hdu.header.set('INSTRUME', aperture.InstrName)
                    DQImage_hdu.header.set('APERTURE', aperture.AperName)
                    DQImage_hdu.header.set('SCA_NUM', int(re.sub("[^0-9]", "", aperture.AperName)))
                    DQImage_hdu.header.set('FILTER', "{0:s}".format(filterName), 'filter')
                    DQImage_hdu.header.set('BUNIT', "unitless", 'brightness units')
                    DQImage_hdu.header.set('exptime', "{0:.3f}".format(exposureTime.value), 'exposure time in sec')
                    DQImage_hdu.header.set('ORIENTAT', "{0:f}".format(pa_aper.deg))
                    DQImage_hdu.header.set('PA_V3', "{0:f}".format(pa_v3.deg))
                    DQImage_hdu.header.set('BG_RATE', "{0:f}".format(bgRate.value), 'sky background rate in ADU/s')
                    DQImage_hdu.header.update(wcs.to_header())

                    hduListOut.append(DQImage_hdu)

                    image_hduNF = fits.ImageHDU((specRate * exposureTime).value, name='TRUE', ver=1)
                    image_hduNF.header.set('TELESCOP', aperture.observatory)
                    image_hduNF.header.set('INSTRUME', aperture.InstrName)
                    image_hduNF.header.set('APERTURE', aperture.AperName)
                    image_hduNF.header.set('SCA_NUM', int(re.sub("[^0-9]", "", aperture.AperName)))
                    image_hduNF.header.set('FILTER', "{0:s}".format(filterName), 'filter')
                    image_hduNF.header.set('BUNIT', str(noisySpec.unit), 'brightness units')
                    image_hduNF.header.set('exptime', "{0:.3f}".format(exposureTime.value), 'exposure time in sec')
                    image_hduNF.header.set('ORIENTAT', "{0:f}".format(pa_aper.deg))
                    image_hduNF.header.set('PA_V3', "{0:f}".format(pa_v3.deg))
                    image_hduNF.header.set('BG_RATE', "{0:f}".format(bgRate.value), 'sky background rate in ADU/s')
                    image_hduNF.header.update(wcs.to_header())

                    hduListOut.append(image_hduNF)

                    hduList = fits.HDUList(hduListOut)

                    hduList.info()

                    outFilename = "test_{0:s}_{1:s}_rollAngle{2:03.0f}_dither{3:d}.fits".format(filterName, aperture.InstrName, rollAngle.deg, ditherN)

                    outFilepath = hostsubdir + 'simdata_prism_galsn/' + outFilename
                    hduList.writeto(outFilepath, overwrite=True)

                    print("Image written to {0:s}".format(outFilename))
                    print("DONE! ELAPSED TIME:", convertTime(time.time() - startTime))

print("ALL DONE! ELAPSED TIME:", convertTime(time.time() - startTimeAll))

sys.exit(0)
