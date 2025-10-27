import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import astropy
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter

import sys
import os
# from tqdm import tqdm
from pprint import pprint
import yaml
import subprocess

home = os.getenv('HOME')
prismdir = home + '/Documents/Roman/PIT/prism/'
hostsubdir = prismdir + 'hostlight_subtraction/'
datadir = hostsubdir + 'simdata_prism_galsn/'
utils_dir = hostsubdir + 'roman-prism-hostsub/utils/'
srccodesdir = hostsubdir + 'roman-prism-hostsub/src/'
sys.path.append(utils_dir)
import romanprism_fast_x1d as oned_utils  # noqa
import get_app_mag as gm  # noqa
from bcolors import bcolors as bc  # noqa
import romanprismdefs as rdef  # noqa


def get_model_init(model, y_fit, x_fit, xhostloc, contamloc):

    # if model == 'sersic':
    #     # Initial guesses
    #     amplitude_init = y_fit.max()
    #     r_init = 10  # half-light radius; pixels
    #     # init sersic index
    #     # 1 is exponential; 4 is de vaucouleurs; 0.5 is gaussian
    #     n_init = 4
    #     # print('Initial guesses for Sersic profile:')
    #     # print('r-eff (half-light rad; pix):', r_init)
    #     # print('n [Sersic index]:', n_init)
    #     model_init = models.Sersic1D(amplitude=amplitude_init, r_eff=r_init,
    #                                  n=n_init)

    # First initialize the host model
    # Initial guess: amplitude=max(y), x_0 at peak, gamma=1, alpha=1.5
    amplitude_init = y_fit.max()
    x_0_init = xhostloc
    gamma_init = 5
    alpha_init = 1.5

    host_init = models.Moffat1D(amplitude=amplitude_init, x_0=x_0_init,
                                gamma=gamma_init, alpha=alpha_init,
                                fixed={'x_0': True},
                                bounds={'amplitude': (0, amplitude_init),
                                        'gamma': (0, 10),
                                        'alpha': (0, 10)})

    # NOw the SN
    # sn_amp_init = y_fit.max() / 2
    # sn_x0_init = 50
    # sn_init = models.Moffat1D(amplitude=sn_amp_init, x_0=sn_x0_init,
    #                           gamma=gamma_init, alpha=alpha_init,
    #                           fixed={'x_0': True},
    #                           bounds={'amplitude': (0, sn_amp_init),
    #                                   'gamma': (0, 10),
    #                                   'alpha': (0, 10)})

    model_init = host_init

    # Now initialize all contaminating galaxies
    for g in range(len(contamloc)):
        gal_amp_init = y_fit.max()
        gal_x0_init = contamloc[g][0]
        gal_init = models.Moffat1D(amplitude=gal_amp_init, x_0=gal_x0_init,
                                   gamma=gamma_init, alpha=alpha_init,
                                   fixed={'x_0': True},
                                   bounds={'amplitude': (0, gal_amp_init),
                                           'gamma': (0, 10),
                                           'alpha': (0, 10)})

        model_init += gal_init

    # Now add a low-order polynomial for the background


    return model_init


def prep_fit(cfg, y, x=None, mask=None):
    """
    Function to prepare x,y arrays for 1D model fitting.
    It does the following things:
    - generate x array if not provided
    - apply any masks given by user
    - ensure valid values in y array by masking NaN values
    - check that there are at least 'numpix_fit_thresh'
      valid pixels to fit. After NaNs are masked and SN+host
      are masked it needs these min number of pix to fit.
    """

    # Generate x array as needed
    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)

    # Apply mask
    donotmask = cfg['donotmask']
    if mask is not None and not donotmask:
        x_fit = np.delete(x, mask)
        y_fit = np.delete(y, mask)
    else:
        x_fit = x
        y_fit = y

    # Additionally mask NaNs
    valid_mask = np.isfinite(y_fit)
    x_fit = x_fit[valid_mask]
    y_fit = y_fit[valid_mask]

    # Also check min len requirement
    prep_flag = True
    numpix_fit_thresh = cfg['numpix_fit_thresh']
    if len(x_fit) < numpix_fit_thresh:
        prep_flag = False

    return x_fit, y_fit, prep_flag


def fit_1d(y_fit, x_fit, contamloc, xhostloc=50, model=None, row_idx=None):
    """
    Fit Moffat or Sersic profiles to 1D data using astropy.

    Parameters:
        y (array-like): 1D data values.
        x (array-like, optional): x values. If None, uses np.arange(len(y)).

    Returns:
        fitted_model: The best-fit astropy model.
    """

    model_init = get_model_init(model, y_fit, x_fit, xhostloc, contamloc)

    fit = fitting.TRFLSQFitter()
    try:
        fitted_model = fit(model_init, x_fit, y_fit)
    except (ValueError, astropy.modeling.fitting.NonFiniteValueError) as e:
        print('\nEncountered exception:', e)
        print(x_fit)
        print(y_fit)
        print('x loc:', xhostloc)
        print('Row index:', row_idx)
        print('\n')

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(x_fit, y_fit, 'o', markersize=4, color='k')
        # ax.set_title('Fitting failed due to non-finite values.\n' +
        #              'Row index: ' + str(row_idx))
        # plt.show()

        # Return a NoneType model that we will check for
        # in the model gen func and skip.
        fitted_model = None
        # sys.exit(1)

    return fitted_model


def get_contiguous_slices(arr, min_length=10):
    """
    Finds the start and end indices of contiguous True
    slices of at least min_length.
    Returns a list of (start, end) tuples (end is inclusive).
    """

    result = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i]:
            start = i
            while i < n and arr[i]:
                i += 1
            end = i - 1
            if end - start + 1 >= min_length:
                result.append((start, end))
        else:
            i += 1

    return result


def get_row_std_objmasked(arr, mask=None):
    if mask is not None:
        arr_fit = np.delete(arr, mask)
        std = np.nanstd(arr_fit)
        return std
    else:
        return np.nanstd(arr)


def get_obj_img_coords(skycoord, wcs):

    objloc = wcs.world_to_pixel(skycoord)
    x = float(objloc[0])
    y = float(objloc[1])

    return (x, y)


def get_sn_host_loc(fname, sciext, snra, sndec, hostra, hostdec):

    header = fits.getheader(datadir + fname, ext=sciext)
    wcs = WCS(header)

    sncoord = SkyCoord(snra, sndec, frame='icrs', unit='deg')
    xsn, ysn = get_obj_img_coords(sncoord, wcs)

    hostcoord = SkyCoord(hostra, hostdec, frame='icrs', unit='deg')
    xhost, yhost = get_obj_img_coords(hostcoord, wcs)

    return xsn, ysn, xhost, yhost


def get_sn_mask(cfg):
    # We actually need to create the masks first
    # mask the SN
    # Note that these indices are relative to the cutout indices.
    # because the cutout was already centered on xsn we can just
    # mask out hte central pixels

    # Oct 2025 update: Now only masking the SN when we
    # have host and multiple contaminant galaxies.

    snmaskpad = cfg['snmaskpad']
    cs_x = cfg['cutoutsize_x']
    # galmaskpad = cfg['galmaskpad']
    sn_mask_idx = np.arange(int(cs_x/2) - snmaskpad,
                            int(cs_x/2) + snmaskpad)
    # mask the galaxy
    # print('Host cutout center idx:', cutout_host_x)
    # gal_mask_idx = np.arange(cutout_host_x - galmaskpad,
    #                          cutout_host_x + galmaskpad)

    # combinedmask = np.union1d(sn_mask_idx, gal_mask_idx)
    # print('combined mask:', combinedmask)
    return sn_mask_idx


def gen_src_list(cfg):

    # Run SExtractor first
    run_sextractor(cfg)

    # Flag to tell downstream function(s) if this function
    # has returned RA,DEC or X,Y
    wcs_coord = False

    # Read the catalog SExtractor generated and get the RA,DEC
    cat_filename = datadir + cfg['cat_filename']
    catheader = ['NUMBER',
                 'X_IMAGE',
                 'Y_IMAGE',
                 'ALPHA_J2000',
                 'DELTA_J2000',
                 'FLUX_AUTO',
                 'FLUXERR_AUTO',
                 'MAG_AUTO',
                 'MAGERR_AUTO']
    cat = np.genfromtxt(cat_filename, dtype=None, names=catheader,
                        encoding='ascii')

    # We're using the X,Y for now because for some reason
    # SExtractor isn't getting the RA,DEC right. It does
    # get teh X,Y correct.
    if wcs_coord:
        allgalc1 = cat['ALPHA_J2000']
        allgalc2 = cat['DELTA_J2000']
    else:
        allgalc1 = cat['X_IMAGE']
        allgalc2 = cat['Y_IMAGE']

    # print('SExtractor coords with SN+HOST included:')
    # print(allgalc1)
    # print(allgalc2)
    # print('SN and HOST coords:')
    # print(xsn, ysn, xhost, yhost)

    # Remove the SN and host from this list. We only want other
    # potential contaminants.
    match_idx_sn = np.argmin(np.sqrt((xsn - allgalc1)**2
                                     + (ysn - allgalc2)**2))
    match_idx_host = np.argmin(np.sqrt((xhost - allgalc1)**2
                                       + (yhost - allgalc2)**2))
    match_idx_list = [match_idx_sn, match_idx_host]

    # remove these indices
    allgalc1 = np.delete(allgalc1, match_idx_list)
    allgalc2 = np.delete(allgalc2, match_idx_list)

    # print('Returning SExtractor coords with SN+HOST excluded:')
    # print(allgalc1)
    # print(allgalc2)

    return allgalc1, allgalc2, wcs_coord


def select_contam_gal(fname, cfg):
    """
    Get contaminating galaxy coordinates given coordinates of all galaxies
    in the direct image and the SN coordinate. This function will run
    an intermediate SExtractor step to detect objects in the direct
    image and return those coordinates.
    This function will return contam gal coords in (x,y) relative to
    the cutout coords.

    Note that for now it is assumed that all gal coords given are
    the undeviated position in the dispersed image (at 1.5um). So
    this function is simply assuming that the dispersed spectrum is
    exactly aligned with the y-direction of the direct image. Effectively
    assuming a PA=0.0 degrees.
    See TODO list for tasks for this function.
    """
    contam_gal_c1, contam_gal_c2, wcs_coord = gen_src_list(cfg)

    # Set up
    ydifflim = 300  # max expected len of SN + contam gal prism spectrum
    ymargin = 50  # padding
    # the x difference should be the size of the galaxy in
    # the cross-dispersion direction and user-provided or
    # determined from teh direct image and PA.
    xdifflim = 50
    xmargin = 100  # padding

    # Also need cutout size
    cs_x = cfg['cutoutsize_x']

    # Get SN coords
    snra = cfg['snra']
    sndec = cfg['sndec']
    # Need these in x,y
    sciext = cfg['sciextnum']
    header = fits.getheader(datadir + fname, ext=sciext)
    wcs = WCS(header)
    sncoord = SkyCoord(snra, sndec, frame='icrs', unit='deg')
    xsn, ysn = get_obj_img_coords(sncoord, wcs)

    # Loop over all objects
    contam_gal_list = []
    # We want the x,y coordinates relative to the cutout but
    # this function can handle inputs of RA, DEC or X, Y (image).
    for i in range(len(contam_gal_c1)):

        # IF we were given RA,DEC we will need the X,Y image coords first
        if wcs_coord:
            galra = contam_gal_c1[i]
            galdec = contam_gal_c2[i]
            # Now convert using SkyCoord to (x,y)
            galcoord = SkyCoord(galra, galdec, frame='icrs', unit='deg')
            xgal, ygal = get_obj_img_coords(galcoord, wcs)
        else:
            xgal = contam_gal_c1[i]
            ygal = contam_gal_c2[i]

        # Now determine if this object is close enough to the SN
        if ((abs(ysn - ygal) <= (ydifflim + ymargin))
                and (abs(xsn - xgal) <= (xdifflim + xmargin))):

            # NOw convert to coordinates relative to the cutout
            xgal_cutout = int(cs_x/2) + int(int(xgal) - xsn)
            cll = cfg['cutoutsize_y_lo']
            ygal_cutout = int(cll) + int(int(ygal) - ysn)

            contam_gal_list.append([xgal_cutout, ygal_cutout])

    return contam_gal_list


def gen_contam_model(cutout, fname, cfg):

    # host_model_to_fit = 'moffat'
    # sn_model_to_fit = 'moffat'

    # ----- Get user config values needed
    start_row = cfg['start_row']
    # You can change end row here for diagnostic purposes
    # The end_row should be set to the end of the cutout
    # but this can be set to the start_row + some other
    # number of rows that you'd like to see fits for
    end_row = cutout.shape[0]  # start_row + 30  # cutout.shape[0]

    # fit thresh
    # sigma_thresh = cfg['sigma_thresh']
    # numpix_fit_thresh = cfg['numpix_fit_thresh']

    # verbosity
    verbose = cfg['verbose']

    # ----- Masks
    # Not considering host or contaminating galaxy masks anymore.
    # All objects fit with compound model simultaneously.
    # However, the SN must be masked. This is to ensure that the SN
    # spectrum is not subtracted out in the final step, which is to
    # get the SN spectrum as the difference between the original data
    # and the contamination model.
    sn_mask_idx = get_sn_mask(cfg)

    # ----- Figure out which objects are to be fit and generate compound model
    """
    For the entire cutout, we consider the objects whose spectra
    might be on that cutout. We don't do this row by row. The fitting
    func can handle the case where an amplitude should be zero for
    an object on a given row.

    Next we construct a compound astropy fitting model
    (i.e., sum of all models for all objects + background) to fit.
    Then proceed to fitting.
    """
    contam_gal_centers = select_contam_gal(fname, cfg)
    print('Contaminating object locations that will be fit',
          '(coordinates relative to cutout):')
    print(contam_gal_centers)

    # ----- loop over all rows
    # Empty array for host model
    contam_model = np.zeros_like(cutout)
    # Dict for params
    contam_fit_params = {}

    for i in range(start_row, end_row):
        if verbose:
            print('Fitting row:', i, end='\r')
        profile_pix = cutout[i]

        # ----- Apply a Savitsky-Golay filter to the data, if user requested
        applysmoothing = cfg['applysmoothingperrow']
        if applysmoothing:
            profile_pix = savgol_filter(profile_pix, window_length=5,
                                        polyorder=2)

        # ----- Skipping criterion
        """
        stdrow = get_row_std_objmasked(profile_pix, mask=None)
        # print('Estimated stddev (with masked SN and host locations)',
        #       'of this row of pixels:', stdrow)
        # contiguous pixels over some threshold
        pix_over_thresh = (profile_pix > sigma_thresh*stdrow)  # boolean
        res = get_contiguous_slices(pix_over_thresh,
                                    min_length=numpix_fit_thresh)
        if not res:
            if verbose:
                print('Row:', i, 'Too few pixels above',
                      sigma_thresh, 'sigma to fit. Skipping.')
            continue
        """

        # ------ Proceed to fitting
        xfit, yfit, pflag = prep_fit(cfg, profile_pix, xarr, mask=sn_mask_idx)
        contam_fit = fit_1d(yfit, xfit, contam_gal_centers,
                            xhostloc=cutout_host_x, row_idx=i)

        # OLD code when SN and HOST were fit separately
        """
        gal_x_fit, gal_y_fit, gal_prep_flag = prep_fit(cfg, profile_pix, xarr)
        if not gal_prep_flag:
            continue
        galaxy_fit = fit_1d(gal_y_fit, gal_x_fit, xloc=cutout_host_x,
                            model=host_model_to_fit, row_idx=i)
        if galaxy_fit is None:
            print('Galaxy fit failed for row:', i)
            continue

        # Fit a moffat profile to the supernova "residual"
        residual_pix = profile_pix - galaxy_fit(xarr)

        # Now mask the galaxy indices
        sn_x_fit, sn_y_fit, sn_prep_flag = prep_fit(cfg, residual_pix, xarr)
        if not sn_prep_flag:
            continue
        moffat_fit = fit_1d(sn_y_fit, sn_x_fit,
                            model=sn_model_to_fit, row_idx=i)
        if moffat_fit is None:
            print('SN fit failed for row:', i)
            continue
        """

        if cfg['showfit']:
            print('----------------')
            print('Contamination fit result:')
            print(contam_fit)

            fig = plt.figure(figsize=(7, 5))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            # plot points and fit
            ax1.plot(xarr, profile_pix, 'o', markersize=4, color='k')
            # ax1.set_yscale('log')
            ax1.plot(xarr, contam_fit(xarr), color='orange',
                     label='Contam fit. Row: ' + str(i))
            ax1.legend(loc=0)

            # plot SN residuals
            ax2.scatter(xarr, profile_pix - contam_fit(xarr), s=5, c='k',
                        label='Residuals after removing contam fit')
            ax2.legend(loc=0)
            # plot SN mask
            ax2.axvspan(sn_mask_idx[0], sn_mask_idx[-1], alpha=0.3,
                        color='gray')

            plt.show()
            fig.clear()
            plt.close(fig)

        # Save the host fit model
        contam_model_row = contam_fit(xarr)
        contam_model[i] = contam_model_row

        # Save individual fit params
        # host_fit_params['Row' + str(i) + '-amp'] = contam_fit.amplitude.value
        # # host_fit_params['Row' + str(i) + '-x0'] = contam_fit.x_0.value
        # host_fit_params['Row' + str(i) + '-gamma'] = contam_fit.gamma.value
        # host_fit_params['Row' + str(i) + '-alpha'] = contam_fit.alpha.value

    return contam_model, contam_fit_params


def update_contam_model(cutout, contam_model, contam_par, cfg):

    iter_flag = False

    # verbosity
    verbose = cfg['verbose']

    # get user config params needed
    start_row = cfg['start_row']

    # ---------
    # Step 1: Check fit params and smooth out host model
    """
    amplitude_arr = []
    gamma_arr = []
    alpha_arr = []
    row_idx_arr = []
    for r in range(start_row, hmodel.shape[0]):
        try:
            amplitude_arr.append(hfit_par['Row' + str(r) + '-amp'])
            gamma_arr.append(hfit_par['Row' + str(r) + '-gamma'])
            alpha_arr.append(hfit_par['Row' + str(r) + '-alpha'])
            row_idx_arr.append(r)
        except KeyError:
            continue

    row_idx_arr = np.array(row_idx_arr)
    amplitude_arr = np.array(amplitude_arr)
    gamma_arr = np.array(gamma_arr)
    alpha_arr = np.array(alpha_arr)

    # Fit a low-order polynomial to the fit params
    # First mask values where we know there isn't host light
    extrarowcutoff = 30
    valid_idx = np.where(row_idx_arr < (start_row + 210 - extrarowcutoff))[0]
    pamp, pamp_cov = np.polyfit(row_idx_arr[valid_idx],
                                amplitude_arr[valid_idx],
                                deg=1, cov=True)
    pgamma, pgamma_cov = np.polyfit(row_idx_arr[valid_idx],
                                    gamma_arr[valid_idx],
                                    deg=3, cov=True)
    palpha, palpha_cov = np.polyfit(row_idx_arr[valid_idx],
                                    alpha_arr[valid_idx],
                                    deg=3, cov=True)

    # get the error on the polynomial fit
    # these are the errors on the fit parameters
    amp_err = np.sqrt(np.diag(pamp_cov))
    gamma_err = np.sqrt(np.diag(pgamma_cov))
    alpha_err = np.sqrt(np.diag(palpha_cov))

    print(pamp)
    print(pgamma)
    print(palpha)
    print(amp_err)
    print(gamma_err)
    print(alpha_err)

    fig = plt.figure(figsize=(7, 3))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    # show the best fit params
    ax1.plot(row_idx_arr, amplitude_arr, 'o', markersize=4, color='k')
    ax1.set_title(hfit_par['model'] + '-Amplitude')
    ax2.plot(row_idx_arr, gamma_arr, 'o', markersize=4, color='r')
    ax2.set_title(hfit_par['model'] + '-Gamma')
    ax3.plot(row_idx_arr, alpha_arr, 'o', markersize=4, color='b')
    ax3.set_title(hfit_par['model'] + '-Alpha')
    # show the fits
    ax1.plot(row_idx_arr, np.polyval(pamp, row_idx_arr), color='gray')
    ax2.plot(row_idx_arr, np.polyval(pgamma, row_idx_arr), color='gray')
    ax3.plot(row_idx_arr, np.polyval(palpha, row_idx_arr), color='gray')

    # also show error on fits # shaded area
    ax1.fill_between(row_idx_arr, np.polyval(pamp, row_idx_arr) - amp_err,
                     np.polyval(pamp, row_idx_arr) + amp_err,
                     color='gray', alpha=0.5)
    ax2.fill_between(row_idx_arr, np.polyval(pgamma, row_idx_arr) - gamma_err,
                     np.polyval(pgamma, row_idx_arr) + gamma_err,
                     color='gray', alpha=0.5)
    ax3.fill_between(row_idx_arr, np.polyval(palpha, row_idx_arr) - alpha_err,
                     np.polyval(palpha, row_idx_arr) + alpha_err,
                     color='gray', alpha=0.5)

    plt.show()

    sys.exit(0)
    """

    # ---------
    # Step 2: find the single rows in the host model
    # that are empty and fill them in with interpolation.
    # first find all zero rows
    zero_row_idxs = []
    for r in range(start_row, contam_model.shape[0]):
        current_row = contam_model[r]
        # test for all zeros
        if not current_row.any():
            zero_row_idxs.append(r)

    if verbose:
        print('\nFilling in single empty rows in the host model.')
        print('Empty rows:', zero_row_idxs)

    # now check for zero rows which have filled rows on either side
    rows_to_fill = []
    for i, idx in enumerate(zero_row_idxs):
        if ((zero_row_idxs[i-1] != (idx - 1))
                and (zero_row_idxs[i+1] != (idx + 1))):
            rows_to_fill.append(idx)

    for row in rows_to_fill:
        avg_row = (contam_model[row-1] + contam_model[row+1]) / 2
        contam_model[row] = avg_row

    if verbose:
        print('Filled in rows:', rows_to_fill)
        print('\n')

    # ---------
    # Step 3: Stack rows where two or more rows aren't fit
    # Find contiguous zero row idxs in the host model
    # Need to convert to boolean array first.
    # This boolean array is True where the host model has a zero row.
    """
    zero_row_bool = np.zeros(hmodel.shape[0], dtype=bool)
    zero_row_bool[zero_row_idxs] = True
    cont_zero_rows = get_contiguous_slices(zero_row_bool, min_length=2)

    if verbose:
        print('\nStacking contiguous zero rows in the host model.')
        print('Contiguous zero row idxs. Tuples of (start, end):')
        print(cont_zero_rows)

    for j in range(len(cont_zero_rows)):
        start, end = cont_zero_rows[j]
        stack_row_idx = np.arange(start, end+1, dtype=int)

        # Skip if rows to stack aren't expected to have host light
        if start > (start_row + 210):
            continue

        if verbose:
            print('Start, end rows for stack:', start, end)

        remove_rows = np.where(stack_row_idx > (start_row + 210))[0]
        if remove_rows.size > 0:
            stack_row_idx = np.delete(stack_row_idx, remove_rows)
            if verbose:
                print('Removing rows that are not expected',
                      'to have host light.')
                print('New stack row idxs:', stack_row_idx)

        # If the number of rows to stack exceeds some number
        # over which the PSF would be expected to change a lot,
        # then we break up the stack into multiple parts.
        stack_row_idx_list = split_indices(stack_row_idx, max_length=8)
        if len(stack_row_idx_list) > 1:
            for st in stack_row_idx_list:
                galaxy_fit, stack_res = fit_stack(cutout, st)
                hmodel[st] = galaxy_fit(xarr)
        else:
            galaxy_fit, stack_res = fit_stack(cutout, stack_row_idx)
            hmodel[stack_row_idx] = galaxy_fit(xarr)
    """

    # ---------
    # Smooth out the contamination model
    for col in range(contam_model.shape[1]):
        current_col = contam_model[:, col]
        newcol = savgol_filter(current_col, window_length=8, polyorder=3)
        contam_model[:, col] = newcol

    return iter_flag, contam_model, contam_par


def fit_stack(cutout, st):
    # Mean stack
    all_rows_stack = cutout[st]
    mean_stack = np.mean(all_rows_stack, axis=0)

    # NOw fit to the stack and replace all zero rows
    # in the host model with this fit
    # We're going to force fit the mean stack regardless of the prep flag
    sn_mask_idx = get_sn_mask(cfg)
    gal_x_fit, gal_y_fit, gal_prep_flag = prep_fit(cfg, mean_stack, xarr,
                                                   mask=sn_mask_idx)
    gfit = fit_1d(gal_y_fit, gal_x_fit, xhostloc=cutout_host_x,
                  model='moffat')

    # Show stack, fit, and data that went into the stack
    if cfg['show_stack_fit']:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.arange(len(mean_stack)), mean_stack,
                   s=4, c='r', label='Stacked data')
        ax.scatter(np.arange(len(mean_stack)),
                   savgol_filter(mean_stack, window_length=4,
                                 polyorder=2),
                   s=6, c='b', label='SG smoothed data')
        ax.plot(xarr, gfit(xarr), color='g', label='Fit result')
        ax.legend(loc=0)
        plt.show()

    return gfit, mean_stack


def split_indices(indices, max_length):
    """
    Splits a continuous array of indices into
    sub-arrays of length <= max_length.

    Parameters:
        indices (array-like): Continuous array of indices.
        max_length (int): Maximum allowed length for each sub-array.

    Returns:
        list of np.ndarray: List of sub-arrays,
        each with length <= max_length.
    """
    indices = np.asarray(indices)
    n = len(indices)
    if n <= max_length:
        sub_arrays = [indices]
    else:
        sub_arrays = []
        for i in range(0, n, max_length):
            sub = indices[i:i+max_length]
            if len(sub) <= max_length:
                sub_arrays.append(sub)

    return sub_arrays


def run_sextractor(cfg, dry_run=False):
    """
    SExtractor is run in an "aggressive deblending" mode --
    this ends up causing larger sources to be deblended more.
    While this is not physically correct, we use the SExtractor
    output to ensure that bright sources within individual galaxies
    such as HII-regions will be fit separately. This helps fit the
    bright contaminating sources better in our direct subtraction.
    It also has the benefit of accouting for spatially varying
    spectra within a galaxy albeit in a crude manner.
    """

    # get the direct image to run on from the config
    dirimgfname = cfg['dirimgfname']
    # change the direct image name to just iclude the sci extention
    dirimgfname = dirimgfname + '[' + str(cfg['sciextnum']) + ']'

    # set the catalog name
    cat_filename = datadir + cfg['cat_filename']
    sextractor_config_fname = datadir + cfg['sextractor_config_fname']

    # Change directory to where teh direct image is
    os.chdir(datadir)

    # print info
    print(f"{bc.GREEN}", "Running SExtractor command:\n",
          "sex", dirimgfname, "-c", sextractor_config_fname,
          "-CATALOG_NAME", os.path.basename(cat_filename),
          f"{bc.RESET}")

    # Use subprocess to call sextractor.
    # The args passed MUST be passed in this way.
    # i.e., args that would be separated by a space on the
    # command line must be passed here separated by commas.
    # It will not work if you join all of these args in a
    # string with spaces where they are supposed to be;
    # even if the command looks right when printed out.
    if not dry_run:
        subprocess.run(['sex', dirimgfname,
                        '-c', sextractor_config_fname,
                        '-CATALOG_NAME',
                        os.path.basename(cat_filename)], check=True)

    # Change directory back to code
    os.chdir(srccodesdir)

    return None


if __name__ == '__main__':

    # TODO list
    print(f"{bc.BLUE}")
    print('\nTODO LIST:')
    print('* NOTE: Automate the cropping later. You will need galaxy\n',
          'half-light radius or the a & b elliptical axes from\n',
          'something like SExtractor to measure the cutout size.')
    print('* NOTE: Using default half-light radius initial guess.\n',
          'This would be better if user provided.')
    print('* NOTE: Using only Sersic/Gaussian profile to fit galaxy.\n',
          'This really should be a model profile convolved with\n',
          'lambda dependent PSF. Convolve Moffat for SN with PSF too.')
    print('* NOTE: Try to fill in NaN values in the observed data?\n',
          'Only if all 8 pixels surrounding a NaN pixel are valid values.')
    print('* NOTE: Figure out how to handle ERR and DQ extensions.')
    print('* NOTE: Show resid hist with SN masked.')
    print('* NOTE: Try grid of sims.')
    print('* NOTE: Add a polynomial fit for the dispersed background.')
    print('* NOTE: Also need code to handle case where a host',
          'galaxy spectrum\n',
          'is much wider than the usual cutout size of 100x300 pix.\n',
          'E.g., This could be a nearby galaxy that covers much of the FoV.')
    print('* NOTE: TODO for contam gal search', '\n',
          '(i) determine effect of PA on contam gal search.', '\n',
          '(ii) how does the contam gal search change',
          'if the trace is not exactly vertical.')
    print('* NOTE: We are using alpha/delta_J2000 for the SExtractor\n',
          'coordinates and ICRS for the sim images. Is this a problem?')
    print('* NOTE: When using SExtractor to find contaminants,\n',
          'we have to exclude the SN+HOST. Write the generic case\n',
          'for multiple or no matches later within gen_src_list(...) later.')
    print('* NOTE: Double check the scaling below to go from effective area\n',
          'units to throughput for the prism and F129 bandpasses.')
    print('* NOTE: Move to testing with HST data once above items are done.\n',
          'One of the tests should be a comparison to what\n',
          'Russell had for Graur et al.')
    print(f"{bc.RESET}")
    print('\n')

    # ==========================
    # Get configuration
    config_flname = 'user_config_1dhostsub.yaml'
    with open(config_flname, 'r') as fh:
        cfg = yaml.safe_load(fh)

    print('Received the following configuration from the user:')
    pprint(cfg)
    # ==========================

    # Read in the effective area curve for the prism
    # these areas are in square meters
    roman_effarea = np.genfromtxt(prismdir + 'Roman_effarea_20201130.csv',
                                  dtype=None, names=True, delimiter=',')
    prism_effarea = roman_effarea['SNPrism'] * 1e4  # cm2
    prism_effarea_wave = roman_effarea['Wave'] * 1e4  # angstroms

    # Read in input spectra
    # We assume that the host and SN input spectra are the same
    # for the entire file name list provided by the user.
    host_spec_fname = datadir + cfg['host_spec_fname']
    sn_spec_fname = datadir + cfg['sn_spec_fname']
    host_input_wav, host_input_flux = np.loadtxt(host_spec_fname, unpack=True)

    sn_input_wav, sn_input_flux = np.loadtxt(sn_spec_fname, unpack=True)
    # this spectrum was scaled when input into the datacube for this sim
    # it is scaled differently depending on the SN mag required for the sim
    # so check this number in the multigalaxysim.ipynb notebook.
    # we should probably save the scaled SN spectrum that was inserted
    # instead of copy-pasting from the NB.
    snsedscaling = 9.485776713818842
    sn_input_flux *= snsedscaling

    fname_list = cfg['fname']
    for fname in fname_list:
        print('\nWorking on file:', fname)

        # Convert SN and host locations from user to x,y
        # Get user coords
        sciextnum = cfg['sciextnum']
        snra = cfg['snra']
        sndec = cfg['sndec']
        hostra = cfg['hostra']
        hostdec = cfg['hostdec']
        xsn, ysn, xhost, yhost = get_sn_host_loc(fname, sciextnum, snra, sndec,
                                                 hostra, hostdec)

        print('x,y SN and x,y host:',
              '{:.4f}'.format(xsn), '{:.4f}'.format(ysn), '  ',
              '{:.4f}'.format(xhost), '{:.4f}'.format(yhost))
        # convert to integer pixels
        row = int(ysn)
        col = int(xsn)
        # print('SN row:', row, 'col:', col)

        # Load image
        prismimg = fits.open(datadir + fname)
        prismdata = prismimg[sciextnum].data

        # Need WCS before getting contaminating obj locations
        """
        wcs = WCS(fits.getheader(datadir + fname, ext=sciextnum))

        # Get coordinates for potential contaminating objects
        contam_gal_ra, contam_gal_dec = gen_src_list(cfg)
        # Now convert to x and y locations of contaminants
        xcontam = []
        ycontam = []
        for k in range(len(contam_gal_ra)):
            cra = contam_gal_ra[k]
            cdec = contam_gal_dec[k]
            ccoord = SkyCoord(cra, cdec, frame='icrs', unit='deg')
            xcc, ycc = get_obj_img_coords(ccoord, wcs)
            xcontam.append(xcc)
            ycontam.append(ycc)
        """

        # get cutout size config
        cs_x = cfg['cutoutsize_x']
        cs_y_lo = cfg['cutoutsize_y_lo']
        cs_y_hi = cfg['cutoutsize_y_hi']

        # Cutout centered on SN loc
        cutout = prismdata[row - cs_y_lo: row + cs_y_hi,
                           col - int(cs_x/2): col + int(cs_x/2)]

        # Get host location within cutout
        cutout_host_x = int(cs_x/2) + int(int(xhost) - col)
        print('Host cutout x loc:', cutout_host_x)

        # plot
        if cfg['showcutoutplot']:
            """
            We like to show both cutouts: direct and dispersed.
            The dispersed cutout is already made. We are just
            going to show the corresponding direct image region
            alongside in the same figure.
            """
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # first plot dispersed image cutout
            cax = ax.imshow(np.log10(cutout), origin='lower',
                            vmin=1.2, vmax=2.6)
            cbar = fig.colorbar(cax)
            cbar.set_label('log(pix val)')

            # plot corresponding area from direct image.



            # plt.show()
            fig.savefig(fname.replace('.fits', '_cutout.png'), dpi=200,
                        bbox_inches='tight')
            fig.clear()
            plt.close(fig)

        # ==========================
        # Now fit the profile
        xarr = np.arange(cs_x)

        # Fit and update
        contam_model, contam_par = gen_contam_model(cutout, fname, cfg)
        # iter_flag, contam_model, host_fit_params = \
        #     update_contam_model(cutout, contam_model, contam_par, cfg)

        # Iterate and test
        """
        num_iter = 1
        max_iter = cfg['max_iter']
        while num_iter < max_iter:
            print('Iteration:', num_iter)
            host_model, host_fit_params = gen_host_model(cutout, cfg)
            iter_flag, host_model = update_host_model(cutout, host_model,
                                                      host_fit_params, cfg)
            if iter_flag:
                num_iter += 1
                continue
            else:
                break
        print('\nDone in', num_iter, 'iterations.\n')
        """

        # ==========================
        # Get the SN spectrum
        recovered_sn_2d = cutout - contam_model

        imghdr = prismimg[sciextnum].header
        wcs = WCS(imghdr)
        # WCS unused if coordtype is 'pix'.
        # The object X, Y are the center of the cutout
        # and the 1.55 micron row index.
        obj_x = int(cs_x/2)
        obj_y = cutout.shape[0] - cs_y_hi
        # print(obj_x, obj_y)
        # Extraction params from config
        one_sided_width = cfg['obj_one_sided_width']
        swav = cfg['start_wav']
        ewav = cfg['end_wav']
        rs, re, cs, ce, specwav = oned_utils.get_bbox_rowcol(obj_x, obj_y, wcs,
                                                             one_sided_width,
                                                             coordtype='pix',
                                                             start_wav=swav,
                                                             end_wav=ewav)
        # print('\n1D-Extraction params:')
        # print('Row start:', rs, 'Row end:', re)
        # print('Col start:', cs, 'Col end:', ce)

        # Collapse to 1D
        # Since we know the location of the SN, we will just
        # use a few pixels around it.
        spec2d = recovered_sn_2d[rs: re + 1, cs: ce + 1]
        sn_1d_spec = np.nanmean(spec2d, axis=1)
        # print('Isolated 2D and 1D SN spec shapes:')
        # print(spec2d.shape)

        # convert to physical units. # flam
        et = cfg['spec_img_exptime']
        sn_1d_spec_phys = oned_utils.convert_to_phys_units(spec2d, specwav,
                                                           'DN',
                                                           prism_effarea_wave,
                                                           prism_effarea,
                                                           subtract_bkg=False,
                                                           spec_img_exptime=et)

        # also try showing what you'd get if you didn't subtract the host
        spec2d_with_host = cutout[rs: re + 1, cs: ce + 1]
        sn_1d_phys_host = oned_utils.convert_to_phys_units(spec2d_with_host,
                                                           specwav, 'DN',
                                                           prism_effarea_wave,
                                                           prism_effarea,
                                                           subtract_bkg=False,
                                                           spec_img_exptime=et)

        # ==========
        # Show all host subtraction
        fig = plt.figure(figsize=(6, 6))

        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.imshow(np.log10(cutout), origin='lower', vmin=1.5, vmax=2.5)
        ax1.set_title('Original image cutout')

        ax2.imshow(np.log10(contam_model), origin='lower', vmin=1.5, vmax=2.5)
        ax2.set_title('File: ' + fname + '\n' + 'Contamination model')

        ax3.imshow(np.log10(recovered_sn_2d), origin='lower',
                   vmin=1.5, vmax=2.5)
        ax3.set_title('SN residual')

        fig.savefig(fname.replace('.fits', '_2dparamfit.png'),
                    dpi=200, bbox_inches='tight')
        fig.clear()
        plt.close(fig)

        # ==========
        # Get the f129 implied apparent magnitude based on the SED
        # that we extracted.
        # first we need to read in the bandpas
        print('Getting AB mag based on extracted 1D spectrum...')

        f129_wav = roman_effarea['Wave'] * 1e4
        scaling = rdef.get_effarea_throughput_scaling()
        f129_thru = roman_effarea['F129'] * scaling

        f129_mag = gm.get_apparent_mag_noredshift(specwav * 1e4,
                                                  sn_1d_spec_phys,
                                                  f129_wav, f129_thru)
        print('Implied F129 ABmag for SN based on 1D spec:',
              '{:.2f}'.format(f129_mag))

        f129_mag_input = gm.get_apparent_mag_noredshift(sn_input_wav,
                                                        sn_input_flux,
                                                        f129_wav, f129_thru)
        print('Implied F129 ABmag for SN input spec:',
              '{:.2f}'.format(f129_mag_input))

        # ==========
        # Now plot input and recovered spectra
        fig = plt.figure(figsize=(6, 4))
        gs = GridSpec(10, 5, hspace=0.05, wspace=0.05,
                      top=0.95, bottom=0.1, left=0.1, right=0.95)
        ax1 = fig.add_subplot(gs[:7, :])
        ax2 = fig.add_subplot(gs[7:, :])

        ax2.set_xlabel('Wavelength [microns]')
        ax1.set_ylabel('Flam [erg/s/cm2/Angstrom]')
        ax2.set_ylabel('Residuals')

        host_smaller_cutout = \
            contam_model[:, cutout_host_x - 25:cutout_host_x + 25]
        host_rec_flux = np.mean(host_smaller_cutout, axis=1)

        sn_input_wav_microns = sn_input_wav/1e4

        ax1.plot(specwav, sn_1d_spec_phys, '-',
                 color='crimson', lw=1.5, label='Recovered SN spec [flam]')
        ax1.plot(sn_input_wav_microns, sn_input_flux, '-',
                 color='mediumseagreen', label='Input SN spec')

        # Also plot the SN spectrum without host contamination subtracted
        # ax1.plot(specwav, sn_1d_phys_host, '-',
        #          color='slategray', lw=1.5,
        #          label='SN spec without host\n' + 'contam. subtracted')

        ax1.legend(loc=0, fontsize=10)

        ylim_specplot = 7.5e-19

        ax1.set_xlim(0.65, 2.0)
        ax1.set_ylim(0, ylim_specplot)

        ax1.set_xticklabels([])

        # plot residuals
        sn_input_flux_inwav = griddata(points=sn_input_wav_microns,
                                       values=sn_input_flux,
                                       xi=specwav)
        spec_resid = ((sn_input_flux_inwav - sn_1d_spec_phys)
                      / sn_input_flux_inwav)

        ax2.scatter(specwav, spec_resid, s=4, c='k')

        ax2.set_xlim(0.65, 2.0)
        ax2.set_ylim(-1.5, 1.5)

        fig.savefig(fname.replace('.fits', '_1dparamfit_phys.png'),
                    dpi=200, bbox_inches='tight')
        # plt.show()

        # Close image
        prismimg.close()

    sys.exit(0)
