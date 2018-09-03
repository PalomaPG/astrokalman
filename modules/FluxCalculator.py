import numpy as np
from astropy.io import fits
import scipy.ndimage as spn


def naylor_photometry(invvar, diff, psf):
    input = diff * invvar
    flux = spn.convolve(input, psf)
    psf2 = psf ** 2
    conv = spn.convolve(invvar, psf2)
    conv[conv == 0] = 0.000001
    var_flux = 1.0/conv
    return flux*var_flux, var_flux


def calc_fluxes(diff_, psf_, invvar_, aflux_):
    diff = fits.open(diff_)[0].data
    psf = np.load(psf_)
    invvar = fits.open(invvar_)[0].data

    invvar[invvar == np.inf] = 0.01
    flux, var_flux = naylor_photometry(invvar, diff, psf)

    if diff_.find('02t') > 0:
        aflux = np.load(aflux_)[0]
        flux = flux/aflux
        var_flux = var_flux/(aflux * aflux)

        var_flux = np.sqrt(var_flux)
        flux[np.isnan(flux)] == .001

    return flux, var_flux, invvar
