# import argparse as ap
from collections import namedtuple
import os

from astropy.io import fits
import numpy as np
from skimage import measure, morphology
import yaml

# from snappl.config import Config

BackgroundResult = namedtuple('BackgroundResult', ['sky',
                                                   'chi2', 'ndof', 'chi2nu'])


class PrismBackground:
    def __init__(self, scifile, objfile=None, cfgfile='background.yaml'):
        # load the config
        # mastercfg = Config.get()
        # self.cfg = mastercfg.value('spectroscopy.prismbackground')

        self.load_config(cfgfile)

        # set some internal parameters
        # if self.cfg.value('dilateregion.perform'):
        #     self.foot = np.ones(self.cfg.value('dilateregion.size'),
        #                         dtype=bool)
        if self.cfg['dilateregion']['perform']:
            self.foot = np.ones(self.cfg['dilateregion']['size'], dtype=bool)

        # load the science data
        okay = self.load_science_data(scifile)

        if not okay:
            return

        # load the object mask
        self.load_object_mask(objfile)

        # compute initial values
        result = self.max_likelihood(self.objmsk)
        self.update_log(1, *result)

        # do the sigma clipping?
        # if self.cfg.value('sigmaclip.perform'):
        if self.cfg['sigmaclip']['perform']:
            result = self.sigma_clip(result)

        # output the sky subtracted image
        self.write_skysub(result)

    def load_config(self, cfgfile):
        self.cfgfile = cfgfile
        with open(self.cfgfile, 'r') as fp:
            self.cfg = yaml.safe_load(fp)

    def update_log(self, *args):
        if self.cfg['verbose']:
            print(*args)

        # update the log here

    def load_science_data(self, scifile, OU25=True):
        self.scifile = scifile
        self.scibase, self.sciext = os.path.splitext(self.scifile)

        if self.sciext in ('.fits',):
            with fits.open(self.scifile, mode='readonly') as hdul:
                if hdul[0].header.get('SKYSUB', False):
                    self.update_log('Background has been subtracted.')
                    return False

                # read the images
                self.sci = hdul[('SCI', 1)].data
                self.unc = hdul[('ERR', 1)].data
                dqa = hdul[('DQ', 1)].data

                # set the good pixel mask
                self.gpx = (np.bitwise_and(dqa, self.cfg['bitmask']) == 0)

                # if the OU25, gotta do something special
                if OU25:
                    exptime = hdul[0].header['EXPTIME']
                    self.sci /= exptime
                    self.unc = np.sqrt(self.unc)/exptime

        elif self.sciext in ('.asdf',):
            raise NotImplementedError(f"ASDF files are not supported")

        else:
            raise ValueError(f"The file type {self.sciext} is invalid")

        # load the model
        self.mod = np.ones_like(self.sci)

        # scale by the noise.  will be useful later!
        self.S = self.sci/self.unc
        self.M = self.mod/self.unc

        return True

    def write_skysub(self, res):
        if self.sciext in ('.fits',):
            with fits.open(self.scifile, mode='update') as hdul:
                modhdu = fits.ImageHDU(data=res.sky*self.mod)
                modhdu.header['EXTNAME'] = 'BACKGROUND'
                modhdu.header['EXTVER'] = 1
                
                hdul[0].header['SKYSUB'] = (True, "Was sky subtracted?")
                hdul[0].header['SKYVAL'] = (res.sky, 'Value of sky subtracted')
                hdul[0].header['SKYCHI2'] = (res.chi2, 'chi2 of fit')
                hdul[0].header['SKYNDOF'] = (res.ndof, '# of degrees of freedom')

                print("GET MORE INFO INTO HEADER")

                hdul[('SCI', 1)].data = self.sci-modhdu.data
                hdul.append(modhdu)

        elif self.sciext in ('.asdf',):
            raise NotImplementedError(f"ASDF files are not supported")

        else:
            raise ValueError(f"The file type {self.sciext} is invalid")

    def load_object_mask(self, objfile):
        self.objfile = objfile
        if isinstance(self.objfile, str):
            # here apply the mask update function
            raise NotImplementedError("Loading an object mask is not supported")

        else:
            self.objmsk = np.zeros_like(self.sci, dtype=bool)

    def max_likelihood(self, objmsk):
        g = np.logical_and(np.logical_not(objmsk), self.gpx)
        S = self.S[g]
        M = self.M[g]
        ndof = np.count_nonzero(g)

        Foo = np.sum(S*S)
        Fot = np.sum(M*S)
        Ftt = np.sum(M*M)

        sky = Fot/Ftt
        chi2 = Foo-sky*Fot

        res = BackgroundResult(sky, chi2, ndof, chi2/ndof)
        return res

    def remove_small(self, msk):

        # label the mask
        labels = measure.label(msk)

        # remove small regions
        labels = morphology.remove_small_objects(labels,
            self.cfg['removesmall']['minsize'])
        # self.cfg.value('removesmall.minsize'))

        return labels != 0

    def dilate_regions(self, msk):
        
        msk = morphology.binary_dilation(msk, footprint=self.foot)            

        return msk

    def update_objmsk(self, msk):
        # remove small object?
        # if self.cfg.value('removesmall.perform'):
        if self.cfg['removesmall']['perform']:
            msk = self.remove_small(msk)

        # dilate the objects?
        # if self.cfg.value('dilateregion.perform'):
        if self.cfg['dilateregion']['perform']:
            msk = self.dilate_regions(msk)

        return msk

    def sigma_clip(self, res):

        itr = 1
        flags = 0

        while not flags:
            # compute residuals
            R = self.S-res.sky*self.M

            # compute sigma clipped
            objmsk = (R > self.cfg['sigmaclip']['nsigma'])
            # objmsk = (R > self.cfg.value('sigmaclip.nsigma'))

            # do the mask updates
            objmsk = self.update_objmsk(objmsk)

            # maximize the likelihood
            new = self.max_likelihood(objmsk)

            # update stopping flags
            if new.sky <= 0:
                flags += 0b0001

            # if itr >= self.cfg.value('sigmaclip.maxiter'):
            if itr >= self.cfg['sigmaclip']['maxiter']:
                flags += 0b0010

            dsky = new.sky-res.sky
            # if np.abs(dsky) < newsky*self.cfg.value('sigmaclip.epsilon'):
            if np.abs(dsky) < new.sky*self.cfg['sigmaclip']['epsilon']:
                flags += 0b0100

            # update counters
            itr += 1
            res = new
            # sky = newsky
            # chi2 = newchi2

            self.update_log(itr, *new)

        return res


def prismbackground():

    datadir = '/Users/baj/Documents/Roman/PIT/prism/hostlight_subtraction/simdata_prism_galsn/nov2025_test/'
    p = PrismBackground(datadir + 'Roman_TDS_simple_model_SNPrism_34649_14.fits')


if __name__ == '__main__':
    prismbackground()
