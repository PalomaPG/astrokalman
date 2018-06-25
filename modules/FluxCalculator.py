import numpy as np
from astropy.io import fits
import scipy.ndimage as spn
import mahotas as mh


class FluxCalculator(object):


    def naylor_photometry(self, invvar, diff):
        """
        Convoluciona flujos. Calcula flujo y su varianza
        :param invvar: float matrix, Varianza de la imagen de diferencia
        :return void:
        """
        input = diff * invvar
        #print 'dimension input 1 @ naylor_photometry %d' % input_1.ndim
        #print 'dimension psf @ naylor_photometry %d' % self.psf.ndim
        if input.ndim != self.psf.ndim:
            return

        self.flux = spn.convolve(input, self.psf)
        psf2 = self.psf ** 2
        convo = spn.convolve(invvar, psf2)
        convo[convo == 0] = 0.000001
        self.var_flux = 1 / convo
        self.flux = self.flux * self.var_flux

    def calc_fluxes(self, science, diff, psf, invvar):

        science = fits.open(science)[0].data
        diff = fits.open(diff)[0].data
        psf = np.load(psf)
        invvar = fits.open(invvar)[0].data

        invvar[invvar == np.inf] = 0.01
        self.naylor_photometry(invvar)