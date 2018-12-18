from modules.utils import cholesky

import numpy as np
########  ########

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


def propagate_func(func, W_m, W_c,  Xs, D=2):
    #Assesses Ys values
    l = 2*D + 1
    Ys = []
    for i in range(l):
        Ys.append(perform(func, Xs[i]))

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


######## NON-LINEAR FUNCTIONS #########

def simple_linear(X, a=100.0):

    flux = a*X[0,:]
    rate_flux = X[1,:]

    return np.array([flux, rate_flux])


##### MAIN ############

if __name__ == '__main__':
   Xs, Wm, Wc = sigma_points(mean_=np.ones((2, 4094, 2046))*0.67, cov_= np.ones((3, 4094, 2046)))
   y_mean, y_cov = propagate_func(simple_linear, Wm, Wc, Xs)
   print(y_mean)
   print(y_cov)