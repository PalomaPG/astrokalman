from modules.utils import cholesky

import numpy as np
########  ########


def sigma_points(mean_, cov_, lambda_,  d=2):
    """

    :param mean_:
    :param cov_:
    :param kappa:
    :param alpha:
    :param beta:
    :param D:
    :return:
    """
    L, U = cholesky((d+lambda_)*cov_)
    X_0 = mean_
    X_1 = mean_ + L[[0, 1], :]
    X_2 = mean_ + L[[1, 2], :]
    X_3 = mean_ - L[[0, 1], :]
    X_4 = mean_ - L[[1, 2], :]

    return list([X_0, X_1, X_2, X_3, X_4])


def unscent_weights(kappa=0, alpha=10**(-3), beta=2.0, D=2):
    lambda_ = (alpha ** 2) * (D + kappa) - D
    W_m = np.zeros(2*D+1)
    W_m[0] = lambda_ / (D + lambda_)
    W_m[1:] = 1/(2*(D + lambda_))
    W_c = np.zeros(2*D+1)
    W_c[0] = W_m[0] + (1 - alpha**2 + beta)
    W_c[1:] = 1/(2*(D + lambda_))
    return W_m, W_c

def perform(func, *args):
    return func(*args)


def propagate_func(func, W_m, W_c,  Xs, *args, D=2 ):
    #Assesses Ys values
    l = int(2*D + 1)
    Ys = []
    for i in range(l):
        Ys.append(perform(func, Xs[i], args))

    y_mean =  np.zeros((2, 4094, 2046))
    for i in range(l):
        y_mean += W_m[i] * Ys[i]

    y_cov = np.zeros((3, 4094, 2046))
    for i in range(l):
        y_diff_0 = (Ys[i][0] - y_mean[0])
        y_diff_2 = (Ys[i][1] - y_mean[1])
        y_cov[0] += W_c[i] * np.power(y_diff_0, 2)
        y_cov[2] += W_c[i] * np.power(y_diff_2, 2)
        y_cov[1] += W_c[i] * (y_diff_0*y_diff_2)

    return y_mean, y_cov


######## LINEAR FUNCTIONS #########

def simple_linear(X, args):
    delta_t = args[0]
    flux = X[0, :]+ delta_t*X[1, :]

    rate_flux = X[1, :]

    return np.array([flux, rate_flux])


def identity(X):
    return X
##### MAIN ############


if __name__ == '__main__':
    Xs= sigma_points(mean_=np.ones((2, 4094, 2046))*0.67, cov_= np.ones((3, 4094, 2046)))
    Wm, Wc = unscent_weights()
    y_mean, y_cov = propagate_func(simple_linear, Wm, Wc, Xs)