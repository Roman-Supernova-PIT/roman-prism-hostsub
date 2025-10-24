from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import ImageNormalize, LogStretch
from astropy.visualization import MinMaxInterval, ZScaleInterval  # noqa

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy import constants
import sys


def dy(w):
    # Dispersion relation copied from Galsim
    dy = ((-81.993865 + 138.367237*(w - 1.0) + 19.348549*(w - 1.0)**2) /
          (1.0 + 1.086447*(w - 1.0) + -0.573797*(w - 1.0)**2))
    return dy


def wave(ypix):
    """
    This is the inverse of the dispersion relation above.
    Write the above formula as:
        (A + B(w-1) + C(w-1)^2)
    y = -----------------------
        (1 + D(w-1) + E(w-1)^2)
    then invert it to get w as a func of y and the constants.
    You can recast the above formula as:
    aw^2 + bw + c = 0; which can be easily solved.
    Here the constants are:
    a = C - yE
    b = B - yD - 2C + 2yE
    c = A - y - B + yD + C - yE
    """

    A = -81.993865
    B = 138.367237
    C = 19.348549
    D = 1.086447
    E = -0.573797

    a = C - ypix*E
    b = B - ypix*D - 2*C + 2*ypix*E
    c = A - ypix - B + ypix*D + C - ypix*E

    wave = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

    return wave


def get_bbox_rowcol(ra, dec, wcs, box_oneside_width, coordtype,
                    start_wav=0.7, end_wav=1.9):

    # Convert to pixels
    # The WCS is only used if coordtype is 'sky'.
    if coordtype == 'sky':
        obj_coord = SkyCoord(ra*u.deg, dec*u.deg, frame='fk5')
        obj_pix = wcs.world_to_pixel(obj_coord)
        # in principle, this same pix in the dispersed image
        # corresponds to 1.55 micron (zero deviation wav)
        # for the object spectrum.
    elif coordtype == 'pix':
        obj_pix = [ra, dec]

    # Now create a 2D bounding box centered on these coords
    # No +1 needed here because these are already 0-indexed.
    # IF you want ds9 based coords (1-indexed) then add 1 here.
    obj_pix_x = obj_pix[0]
    obj_pix_y = obj_pix[1]

    start_y = np.floor(obj_pix_y + dy(start_wav)).astype(int)
    end_y = np.ceil(obj_pix_y + dy(end_wav)).astype(int)
    start_x = int(obj_pix_x - box_oneside_width)
    end_x = int(obj_pix_x + box_oneside_width)

    # convert pix to numpy row, col
    # No need for any +-1 here because these are already
    # 0-indexed from world_to_pix
    rs = start_y  # row start
    re = end_y  # row end
    cs = start_x  # col start
    ce = end_x  # col end

    # Also return the wavelengths
    # Need +1 to make sure last coord is included.
    ypix_arr = np.arange(start_y - obj_pix_y,
                         end_y - obj_pix_y + 1).astype(int)
    specwav = wave(ypix_arr)

    return rs, re, cs, ce, specwav


def convert_to_phys_units(arr, wav, arr_unit,
                          prism_effarea_wave, prism_effarea,
                          subtract_bkg=False,
                          spec_img_exptime=1000):
    """
    This func will convert the provided array (2D spectrum)
    to physical units (flux density).

    IT expects to get:
    * arr: the 2D spectrum array
    * wav: the associated wavelengths. It WILL ASSUME
    for now that each slice (in x direction) has a constant wav
    associated with it, i.e., the prism disperses completely
    straight in the y direction.
    -- Expected units of microns.

    * arr_unit: units of the 2D spectral array. Either DN or
    photons.

    """

    # Check array units
    """
    if arr_unit == 'DN':
        return
        # elif arr_unit == 'photons':
    else:
        print('Array unit not understood.',
              'Can only accept DN or photons for now.')
        sys.exit(0)
    """

    # Define some constants
    c_ang = constants.speed_of_light * 1e10  # A/s; default in m/s
    planck_cgs = constants.Planck * 1e7  # joules to ergs

    # Check shape of input 2D array
    # We want to ensure more rows than cols -- i.e.,
    # that the spectrum was dispersed along the y axis.
    assert arr.shape[0] > arr.shape[1]
    assert wav.shape[0] == arr.shape[0]

    num_pix_slice = arr.shape[1]

    # For background subtraction
    if subtract_bkg:
        (rs_bkg, re_bkg,
         cs_bkg, ce_bkg, _specwav) = get_bbox_rowcol(bkg_ra, bkg_dec,
                                                     wcs,
                                                     bkg_one_sided_width,
                                                     coordtype='sky')
        bkg_2dbox = img_2d_scidata[rs_bkg: re_bkg, cs_bkg: ce_bkg]
        bkg = np.mean(bkg_2dbox)
        print('\nBackground mean level:', bkg, '\n')
    else:
        bkg = 0

    # Convert to flux density units
    # convert to energy rate first
    pix_energies = np.zeros(len(wav))

    spec_1dphys = np.zeros(len(wav))

    # I'm doing this in an explicit for loop because
    # later on we will probably not have a constant wavelength
    # along the x direction and will have to do this
    # explicity instead of numpy vectorized operations.

    for i in range(len(wav)-1):

        # sum the photons along the slice
        total_photons_in_slice = np.sum(arr[i])

        # subtract background
        total_photons_in_slice -= (bkg*num_pix_slice)

        # convert to an energy
        pix_cen_wav = wav[i] * 1e4  # need this in angstroms here
        pix_energies[i] = (total_photons_in_slice * planck_cgs * c_ang
                           / pix_cen_wav)

        # now convert to energy rates
        pix_energies[i] /= spec_img_exptime

        # now convert to energy rate density
        # the lambda range here must be in angstroms
        lam_rng = (wav[i+1] - wav[i]) * 1e4
        # lam_rng = wave(ypix + 0.5)*1e4 - wave(ypix - 0.5)*1e4
        pix_energies[i] /= lam_rng

        # Conversion to flux density
        pix_eff_area = get_pix_eff_area(pix_cen_wav,
                                        prism_effarea_wave, prism_effarea)
        # since we are also summing along the x axis
        # we are summing more than one pixel
        pix_eff_area *= num_pix_slice

        spec_1dphys[i] = pix_energies[i] / pix_eff_area

        # print(i, '  ', '{:.2f}'.format(pix_cen_wav), '  ',
        #       '{:.2e}'.format(pix_energies[i]), '  ',
        #       '{:.2f}'.format(lam_rng), '  ',
        #       '{:.2f}'.format(pix_eff_area), '  ',
        #       '{:.2e}'.format(spec_1dphys[i]))

    if subtract_bkg:
        return spec_1dphys, bkg_2dbox, bkg
    else:
        return spec_1dphys


def get_pix_eff_area(lam, prism_effarea_wave, prism_effarea):
    lam_idx = np.argmin(np.abs(prism_effarea_wave - lam))
    return prism_effarea[lam_idx]


if __name__ == '__main__':

    # =============================
    # User inputs
    # =============================
    plotfigure = True
    save1d = True
    save2d = False

    subtract_bkg = False  # turn off if using noiseless image

    spec_img_exptime = 1000  # seconds

    # Extraction box widths
    # Needs odd numbers
    obj_one_sided_width = 5
    bkg_one_sided_width = 5

    # Wavelengths for extraction
    user_start_wav = 0.8  # microns
    user_end_wav = 1.9  # microns

    # Source location
    obj_x = 827.568
    obj_y = 687.048

    # Enter coords of a location in direct image
    # such that a ~200 Ypix x 20 Xpix box centered
    # on this location will not have a spectrum in it.
    # This will be used to estimate the background.
    bkg_ra = 7.55221080
    bkg_dec = -44.80798231

    # Path to data
    datadir = '/Users/baj/Documents/Roman/prism_quick_reduction/testdata/'

    # Image names
    # noised 2D image
    imgfname = 'Roman_TDS_truth_SNPrism_0_5_bj.fits'
    # truth 2D image
    # imgfname = 'Roman_TDS_truth_SNANAonly_SNPrism_36462_5.fits'
    # Direct image
    # dirimgname = imgfname.replace('SNPrism', 'H158')
    # dirimgname = 'Roman_TDS_simple_model_H158_36462_5.fits'

    truespec_fname = datadir + 'M82_starburst_template.sed'
    h5spec_fname = datadir + 'simgal_sfg_h5.txt'

    # === Stuff below for scaling true spectrum
    scaletruespec = True
    truespec_scaleflam = 1e-15  # will make truespec flam match this number
    # will make truespec flam match above number at this wav
    truespec_scalewav = 1.2  # microns

    # =============================
    # User inputs above this line
    # =============================

    # First read in the effective area curve for the prism
    # these areas are in square meters
    roman_effarea = np.genfromtxt(datadir + 'Roman_effarea_20201130.csv',
                                  dtype=None, names=True, delimiter=',')

    prism_effarea = roman_effarea['SNPrism'] * 1e4  # cm2
    prism_effarea_wave = roman_effarea['Wave'] * 1e4  # angstroms

    # output todo:
    # build in a hook for reading error extension and error propagation

    # Set up image paths
    # img_2d = fits.open(datadir + imgfname)
    # img_direct = fits.open(datadir + dirimgname)

    # img_2d_scidata = fits.getdata(datadir + imgfname, extname='SCI')
    img_2d_scidata = fits.getdata(datadir + imgfname)
    # dirhdr = img_direct[0].header

    # Read in true spectrum
    truespec = np.genfromtxt(truespec_fname,
                             dtype=None, names=['wave', 'flux'],
                             encoding='ascii', skip_header=3)
    truespec_wav = truespec['wave']

    speed_of_light_ang = 3e18  # A/s

    # truespec_fnu = truespec['flux']
    # truespec_flam = truespec_fnu * speed_of_light_ang / truespec_wav**2
    truespec_flam_scaled = truespec['flux']

    # now convert the true wav to microns
    truespec_wav /= 1e4

    # Scale true spec
    if scaletruespec:
        scalewav_idx = np.argmin(np.abs(truespec_wav - truespec_scalewav))
        scalefac = truespec_scaleflam / truespec_flam_scaled[scalewav_idx]
        truespec_flam = truespec_flam_scaled * scalefac

    # Read in spec in H5 file
    h5spec = np.genfromtxt(h5spec_fname, dtype=None,
                           names=True, encoding='ascii')
    h5spec_wav = h5spec['wav'] / 1e4
    h5spec_flam = h5spec['flam']

    # =========
    # Get object 2D and 1D spectrum
    imghdr = fits.getheader(datadir + imgfname)
    wcs = WCS(imghdr)
    rs, re, cs, ce, specwav = get_bbox_rowcol(obj_x, obj_y, wcs,
                                              obj_one_sided_width,
                                              coordtype='pix',
                                              start_wav=user_start_wav,
                                              end_wav=user_end_wav)
    spec2d = img_2d_scidata[rs: re+1, cs: ce+1]
    # Need +1 to make sure last coord is included.

    # make sure that the x width of the extraction box is as expected
    assert spec2d.shape[1] == 2*obj_one_sided_width + 1

    # --- convert to physical units
    if subtract_bkg:
        spec1d_phys, bkg_2dbox, bkg = convert_to_phys_units(arr=spec2d,
                                                            wav=specwav,
                                                            arr_unit='DN')
    else:
        spec1d_phys = convert_to_phys_units(arr=spec2d, wav=specwav,
                                            arr_unit='DN')
        bkg = 0
    rawspec = np.sum(spec2d, axis=1)

    # Write to file
    with open('romanprism_x1d_test.txt', 'w') as fh:
        fh.write('#  wav_micron  flam' + '\n')
        for i in range(len(spec1d_phys)):
            fh.write('{:.4f}'.format(specwav[i]) + '  ')
            fh.write('{:.2e}'.format(spec1d_phys[i]))
            fh.write('\n')

    # Figure
    if plotfigure:
        fig = plt.figure(figsize=(9, 5))
        gs = GridSpec(nrows=10, ncols=10)
        ax1 = fig.add_subplot(gs[:, :1])
        ax2 = fig.add_subplot(gs[:, 1:2])
        ax3 = fig.add_subplot(gs[:5, 3:])
        ax4 = fig.add_subplot(gs[5:, 3:])

        ax3.set_ylabel('flam [erg/s/cm2/A]', fontsize=13)
        ax4.set_xlabel('Wavelength [Microns]', fontsize=13)
        ax4.set_ylabel('Raw Photons', fontsize=13)

        if subtract_bkg:
            ax1.imshow(bkg_2dbox, origin='lower')
        # turn off spines and ticklabels
        ax1.set_axis_off()

        # Log stretch for 2D spectrum
        spec2d -= bkg
        img_vmin = np.min(spec2d)
        img_vmax = np.max(spec2d)
        norm = ImageNormalize(vmin=img_vmin, vmax=img_vmax,
                              stretch=LogStretch())
        ax2.imshow(spec2d, origin='lower', norm=norm,
                   cmap='viridis')
        # turn off spines and ticklabels
        ax2.set_axis_off()

        # Extracted spectrum and injected spectrum
        ax3.plot(specwav, spec1d_phys, color='k', label='1d spectrum')
        ax3.plot(truespec_wav, truespec_flam, color='crimson',
                 label='input spectrum high-res', lw=1.5)
        # ax3.plot(h5spec_wav, h5spec_flam, color='orchid',
        #          label='input spectrum in h5', lw=1.5)

        ax4.plot(specwav, rawspec, color='k')

        # overplot flat flam spec
        # flat_lam = np.arange(start_wav, end_wav + 10.0, 5.0)
        # flat_flam =
        # ax3.plot(flat_lam, flat_flam, color='r',
        #          label='flat flam spec', lw=1.0)

        # Limits
        ax3.set_ylim(8e-17, 4e-15)

        ax3.set_xlim(user_start_wav, user_end_wav)
        ax4.set_xlim(user_start_wav, user_end_wav)

        ax3.set_yscale('log')

        # Legend
        ax3.legend(loc='upper right', fontsize=11)

        # Ticklabels
        ax3.set_xticklabels([])

        if subtract_bkg:
            fig.savefig('romanprism_testsn_1dspec.pdf', dpi=200,
                        bbox_inches='tight')
        else:
            fig.savefig('romanprism_testsn_1dspec_nobkg.pdf', dpi=200,
                        bbox_inches='tight')

    sys.exit(0)
