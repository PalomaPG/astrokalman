import numpy as np


def simple_linear(X, a=100.0):

    flux = a*X[0,:]
    rate_flux = X[1,:]

    return np.array([flux, rate_flux])