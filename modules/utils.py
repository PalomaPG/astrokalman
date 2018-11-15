import numpy as np
from astropy.io import fits
import scipy.ndimage as spn
import mahotas as mh


import matplotlib.pyplot as plt

from modules.unscented_functions import *

def naylor_photometry(invvar, diff, psf):
    """

    :param invvar:
    :param diff:
    :param psf:
    :return:
    """
    input = diff * invvar
    flux = spn.convolve(input, psf)
    psf2 = psf ** 2
    conv = spn.convolve(invvar, psf2)
    conv[conv == 0] = 0.000001
    var_flux = 1.0/conv
    return flux*var_flux, var_flux


def calc_fluxes(diff_, psf_, invvar_, aflux_):
    """

    :param diff_:
    :param psf_:
    :param invvar_:
    :param aflux_:
    :return:
    """
    diff = fits.open(diff_)
    psf = np.load(psf_)
    invvar = fits.open(invvar_)
    invvar[0].data[invvar[0].data == np.inf] = 0.01
    flux, var_flux = naylor_photometry(invvar[0].data, diff[0].data, psf)
    diff.close()
    invvar.close()

    if diff_.find('02t') > 0:
        aflux = np.load(aflux_)[0]
        flux = flux/aflux
        var_flux = var_flux/(aflux * aflux)

    var_flux = np.sqrt(var_flux)
    flux[np.isnan(flux)] == .001

    return flux, var_flux


def subsampled_median(image, image_size,  sampling):
    """
    :param image: float matrix, input image
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


def get_bin_decomp(num, o, RD, n):
    """
    :param num:
    :param o:
    :param RD:
    :param n:
    :return:
    """
    flags_stats_gr = np.ones(n) * (-1)
    for i in range(n):
        if (num & 1) == 1:
            flags_stats_gr[i] = i
        num = num >> 1
    plt.clf()
    plt.hist(flags_stats_gr, bins=n, range=[0, n-1], align='mid')
    plt.title('Alert freq')
    plt.xlabel('Alert')
    plt.ylabel('Frequency')
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

def mask_and_dilation(mask_path):
    """

    :param mask_path:
    :return:
    """
    mask = fits.open(mask_path)[0].data > 0
    dil_mask = mh.dilate(mask > 0, Bc=np.ones((5, 5), dtype=bool))
    return mask, dil_mask


def median_rejection_calc(median_rejection, accum_median_flux, accum_med_flux_depth, flux, mjd_index):
    """

    :param median_rejection:
    :param accum_median_flux:
    :param accum_med_flux_depth:
    :param flux:
    :param mjd_index:
    :return:
    """
    if mjd_index<accum_med_flux_depth:
        accum_median_flux[mjd_index, :] = flux
    elif mjd_index == accum_med_flux_depth:
        median_rejection = np.median(accum_median_flux, 0) > 1500.0

    return median_rejection, accum_median_flux


def sigma_points(mean_, cov_,
                 kappa=0, alpha=10**(-3),
                 beta = 2.0, D=2):
    """

    :param mean_:
    :param cov_:
    :param kappa:
    :param alpha:
    :param beta:
    :param L:
    :return:
    """

    lambda_ = (alpha**2)*(D + kappa) - D

    L, U = cholesky((D+lambda_)*cov_)
    X_0 = mean_
    X_1 = mean_ + L[[0,1], :]
    X_2 = mean_ + L[[1,2], :]
    X_3 = mean_ - L[[0,1], :]
    X_4 = mean_ - L[[1,2], :]

    W_m = np.zeros(5)
    print(lambda_)
    print(D + lambda_)
    W_m[0] = lambda_ / (D + lambda_)
    W_m[1:] = 1/(2*(D + lambda_))

    W_c = np.zeros(5)
    W_c[0] = lambda_/(D+lambda_) + (1 - alpha**2 + beta)
    W_c[1:] = 1/(2*(D + lambda_))

    return list([X_0, X_1, X_2, X_3, X_4]), W_m, W_c


def perform(func, *args):
    return func(*args)


def propagate_func(func, W_m, W_c,  Xs):
    #Assesses Ys values
    Ys = []
    for i in range(len(Xs)):
        Ys.append(perform(func, Xs[i]))

    y_mean =  0
    for i in range(len(Ys)):
        y_mean += W_m[i] * Ys[i]

    return y_mean



if __name__ == '__main__':
    Xs, Wm, Wc = sigma_points(mean_=np.zeros((2, 4094, 2046)), cov_= np.ones((3, 4094, 2046)))
   #propagate_func(simple_linear, Wm, Wc, Xs)




