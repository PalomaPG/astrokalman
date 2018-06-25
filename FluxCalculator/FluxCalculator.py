import numpy as np
from astropy.io import fits
import scipy.ndimage as spn
import mahotas as mh


class FluxCalculator(object):

    def calc_fluxes(self, data):

        science = fits.open(self.data['science'][o])[0].data
        diff = fits.open(self.data['diff'][o])[0].data
        psf = np.load(self.data['psf'][o])
        invvar = fits.open(self.data['invVAR'][o])[0].data
