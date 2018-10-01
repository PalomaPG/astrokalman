import numpy as np
from astropy.io import fits
import scipy.ndimage as spn

import matplotlib.pyplot as plt


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

    return flux, var_flux


def subsampled_median(image, image_size,  sampling):
    """
    Obtiene la mediana de subimagenes de image
    :param image: float matrix, Imagen de entrada
    :param sampling: int,
    :return:
    """
    size1 = image_size[0]
    size2 = image_size[1]
    margin = 100
    yAxis = range(margin, size1 - margin, sampling)
    xAxis = range(margin, size2 - margin, sampling)
    sampled_image = np.zeros((len(yAxis), len(xAxis)))
    x = 0
    for i in yAxis:
        y = 0
        for j in xAxis:
            sampled_image[x, y] = image[i, j]
            y += 1
        x += 1
    return np.median(sampled_image)


def get_bin_decomp(self, num, o, RD, n):
    flags_stats_gr = np.ones(n) * (-1)
    if num == 0 :
        print('Candidato ideal...epoch: %d' % o)
    for i in range(n):
        if (num & 1) == 1:
            print('Group flags SN: %d' % i)
            flags_stats_gr[i] = i
        num = num >> 1
    plt.clf()
    plt.hist(flags_stats_gr, bins=n, range=[0, n-1], align='mid')
    plt.title('Frecuencia alertas en SN')
    plt.xlabel('Alerta')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.savefig(RD.filter_type + '_' + str(RD.SN_index) + '_' + str(o) + '_' +str(RD.flux_thres) + '.png')


def cholesky(P):
    """
    :param P:
    :return:
    """
    # Cholesky decomposition
    L = np.zeros(P.shape)
    L[0, :] = np.sqrt(P[0, :])
    L[1, :] = P[1, :] / L[0, :]
    L[2, :] = np.sqrt(P[2, :] - np.power(L[1, :], 2))
    # Include inversion
    inv_L = np.ones(L.shape)
    inv_L[1, :] = -L[1, :]
    inv_L[[0, 1], :] = inv_L[[0, 1], :] / L[0, :]
    inv_L[[1, 2], :] = inv_L[[1, 2], :] / L[2, :]
    return L, inv_L


def image_stats(image, outlier_percentage=2.0):
    """
    :param image:
    :param outlier_percentage:
    :return:
    """
    vector = np.reshape(image, -1)
    max_range = np.mean(np.abs(np.percentile(vector, [outlier_percentage, 100.0 - outlier_percentage])))
    vector = vector[vector < max_range]
    vector = vector[vector > -max_range]
    return np.mean(vector), np.std(vector), vector