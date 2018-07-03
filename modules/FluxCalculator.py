import numpy as np
from astropy.io import fits
import scipy.ndimage as spn
import mahotas as mh


class FluxCalculator(object):

    def __init__(self, image_size, accum_neg_flux_depth =4, accum_med_flux_depth=3):
        self.accum_neg_flux_depth = accum_neg_flux_depth
        self.accum_med_flux_depth = accum_med_flux_depth
        self.image_size = image_size

    def naylor_photometry(self, invvar, diff, psf):
        input = diff * invvar
        flux = spn.convolve(input, psf)
        psf2 = psf ** 2
        conv = spn.convolve(invvar, psf2)
        conv[conv == 0] = 0.000001
        var_flux = 1.0/conv
        return flux*var_flux

    def calc_fluxes(self, science_, diff_,
                    psf_, invvar_, aflux_,
                    flux, var_flux):
        science = fits.open(science_)[0].data
        diff = fits.open(diff_)[0].data
        psf = np.load(psf_)
        invvar = fits.open(invvar_)[0].data

        invvar[invvar == np.inf] = 0.01
        self.naylor_photometry(invvar)

        if diff_.find('02t') > 0 :
            aflux = np.load(aflux_)[0]
            flux = flux/aflux
            var_flux = var_flux/(aflux * aflux)

        var_flux = np.sqrt(var_flux)
        flux[np.isnan(flux)] == .001

        '''
        if o < self.accum_med_flux_depth:
            self.accum_median_flux[o, :] = self.flux
        elif o == self.accum_med_flux_depth: # Cubo de booleans
            self.median_rejection = np.median(self.accum_median_flux, 0) > 1500.0
        '''
