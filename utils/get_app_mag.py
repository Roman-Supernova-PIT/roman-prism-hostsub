import numpy as np
from scipy.interpolate import griddata
# Cosmology
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

speed_of_light_ang = 3e18  # angstroms per second


def filter_conv(filter_wav, filter_thru, spec_wav, spec_flam):

    # Ensure that the supplied spectrum does not have NaNs
    spec_valid_idx = np.isfinite(spec_flam)
    # print('Spec valid idx:', spec_valid_idx)

    # First grid the spectrum wavelengths to the filter wavelengths
    spec_on_filt_grid = griddata(points=spec_wav[spec_valid_idx],
                                 values=spec_flam[spec_valid_idx],
                                 xi=filter_wav)
    # print('Spec on filt grid:', spec_on_filt_grid)

    # Remove NaNs again. This is probably redundant.
    valid_idx = np.isfinite(spec_on_filt_grid)

    filter_wav = filter_wav[valid_idx]
    filter_thru = filter_thru[valid_idx]
    spec_on_filt_grid = spec_on_filt_grid[valid_idx]

    # print('Spec on filt grid [flam]:', spec_on_filt_grid)

    # Now do the two integrals
    num = np.trapz(y=spec_on_filt_grid * filter_thru, x=filter_wav)
    den = np.trapz(y=filter_thru, x=filter_wav)
    # print('num:', num)
    # print('den:', den)

    filter_flux = num / den

    return filter_flux


def get_apparent_mag_redshift(redshift, sed_lam, sed_llam,
                              lam_pivot=10552.0, band=None):
    """
    This func will redshift the provided SED and convolve it
    with a bandpass to give an apparent AB magnitude.
    Expects to get:
      redshift -- float
      sed_lam -- wavelengths in A
      sed_llam -- luminosity density in erg/s/A
      lam_pivot -- pivot wav in Angstroms.
                   defaults to Roman WFI/F105W
      band -- numpy record array of the bandpass
              by reading in the bandpass txt file through
              np.genfromtxt(...)
              Expects two cols: wavelength[A] and throughput
              with col names: 'wav' and 'thru'
    """

    # First redshift the provided SED
    lam_obs = sed_lam * (1 + redshift)

    dl_mpc = cosmo.luminosity_distance(redshift).value
    mpc2cm = 3.086e24
    dl = dl_mpc * mpc2cm
    flam = sed_llam / (4 * np.pi * dl * dl * (1 + redshift))

    # Now convolve with the provided filter
    flam_conv = filter_conv(band['wav'], band['thru'], lam_obs, flam)

    # Convert to fnu and get AB mag
    # lam_pivot =

    fnu_conv = lam_pivot**2 * flam_conv / speed_of_light_ang
    appmag = -2.5 * np.log10(fnu_conv) - 48.6

    return appmag


def get_apparent_mag_noredshift(spec_lam_obs, spec_flam_obs,
                                band_wav, band_thru,
                                lam_pivot=12930):
    """
    Essentially the same function as above but this one assumes
    you already have a redshifted SED and are trying to figure out
    the magnitude that would have in a given bandpass.

    Note the difference in units for the provided SED between this
    function and the one above. Also note the different way the bandpass
    is passed and the different default for pivot wavelength.

    Example use case: you extracted a 1D spectrum and want to know what the
    apparent magnitude of this 1D spectrum would be through a given
    filter.

    Expected args:
    spec_lam_obs -- Observed spectrum wav in Angstroms
    spec_flam_obs -- Observed spectrum flux in physical units of fnu;
                     erg/s/cm2/A
    lam_pivot -- pivot wav in Angstroms; defaults to WFI/F129W
    band_wav -- bandpass wav in ang
    band_thru -- bandpass throughput
    """

    # Convolve with the provided filter
    flam_conv = filter_conv(band_wav, band_thru, spec_lam_obs, spec_flam_obs)

    fnu_conv = lam_pivot**2 * flam_conv / speed_of_light_ang
    appmag = -2.5 * np.log10(fnu_conv) - 48.6

    return appmag
