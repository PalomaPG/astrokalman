from modules.utils import cholesky

import numpy as np
########  ########


def sigma_points(mean_, cov_, lambda_,  N=2):
    """

    :param mean_:
    :param cov_:
    :param kappa:
    :param alpha:
    :param beta:
    :param D:
    :return:
    """
    L, U = cholesky((N+lambda_)*cov_)
    #L, U = cholesky(cov_)

    X_0 = mean_
    X_1 = mean_ + L[[0, 1]]
    X_2 = mean_ + L[[1, 2]]
    X_3 = mean_ - L[[0, 1]]
    X_4 = mean_ - L[[1, 2]]

    return list([X_0, X_1, X_2, X_3, X_4])


def unscent_weights(kappa=0, alpha=10**(-3), beta=2.0, N=2):
    lambda_ = (alpha ** 2) * (N + kappa) - N
    W_m = np.zeros(2*N+1)
    W_m[0] = lambda_ / (N + lambda_)
    W_m[1:] = 1/(2*(N + lambda_))
    W_c = np.zeros(2*N+1)
    W_c[0] = W_m[0] + 1 - alpha**2 + beta
    W_c[1:] = 1/(2*(N + lambda_))
    return W_m, W_c

def perform(func, *args):
    return func(*args)


def propagate_func_pred(func, W_m, W_c,  Xs, u, Q, delta_t, args, D=2, image_size=(4094, 2046)):
    #Assesses Ys values

    l = int(2*D + 1)
    Ys = []
    for i in range(l):
        Ys.append(perform(func, Xs[i], [delta_t] + args))

    y_mean =  np.zeros((tuple([2]) + image_size))

    for i in range(l):
        y_mean += W_m[i] * Ys[i]


    y_cov = np.zeros(tuple([3]) + image_size)
    for i in range(l):
        y_diff_0 = (Ys[i][0] - y_mean[0])
        y_diff_2 = (Ys[i][1] - y_mean[1])
        y_cov[0] += W_c[i] * np.power(y_diff_0, 2)
        y_cov[2] += W_c[i] * np.power(y_diff_2, 2)
        y_cov[1] += W_c[i] * (y_diff_0*y_diff_2)

    return y_mean+u, y_cov+Q


def propagate_func_corr(func, W_m, W_c, Xs, delta_t, args, D=2, image_size=(4094, 2046), mean=True):

    l = int(2*D + 1)

    Ys = []
    for i in range(l):
        Ys.append(perform(func, Xs[i], [delta_t] + args))

    y_mean = np.zeros((tuple([2]) + image_size))

    for i in range(l):
        y_mean += W_m[i] * Ys[i]

    if not mean:
        y_cov = np.zeros(tuple([3]) + image_size)

        for i in range(l):
            y_diff_0 = (Ys[i][0] - y_mean[0])
            y_diff_2 = (Ys[i][1] - y_mean[1])
            y_cov[0] += W_c[i] * np.power(y_diff_0, 2)
            y_cov[2] += W_c[i] * np.power(y_diff_2, 2)
            y_cov[1] += W_c[i] * (y_diff_0 * y_diff_2)
        return  y_cov

    return y_mean


#State - measurement
def cross_covariance(f_func, h_func, W_c, Xs, Ys, delta_t, f_args, h_args, pred_state, pred_z,
                     D=2, image_size=(4094, 2046)):
    l = int(2*D + 1)

    CCM = np.zeros(tuple([4]) + image_size)

    for i in range(l):
        F = (perform(f_func, Xs[i], [delta_t] + f_args)-pred_state)
        H = (perform(h_func, Ys[i], [delta_t] + h_args)-pred_z)
        CCM[0] += W_c[i]*F[0] * H[0]
        CCM[3] += W_c[i]*F[1] * H[1]
        CCM[1] += W_c[i]*F[0] * H[1]
        CCM[2] += W_c[i]*F[1] * H[0]

    return CCM

#Optimal gain
def optimal_gain(C, S, image_size=(4094, 2046)):

    K = np.zeros(tuple([4])+image_size)
    print(K.shape)

    alpha = (S[0]*S[2] - S[1]*S[1])**(-1)
    K[0] = alpha*(C[0]*S[2]-C[1]*S[1])
    K[1] = alpha*(-C[0]*S[1]+C[1]*S[0])
    K[2] = alpha*(C[2]*S[2]-C[3]*S[1])
    K[3] = alpha*(-C[2]*S[1]+C[3]*S[0])

    return K


def get_KSKt_product(K,S, image_size):

    KS = np.zeros(tuple([4])+image_size)
    KS[0] = K[0]*S[0] + K[1]*S[1]
    KS[1] = K[0]*S[1] + K[1]*S[2]
    KS[2] = K[2]*S[0] + K[3]*S[1]
    KS[3] = K[2]*S[1] + K[3]*S[2]

    KSKt = np.zeros(tuple([3])+image_size)
    KSKt[0] = KS[0]*K[0] + KS[1]*K[1]
    KSKt[1] = KS[0]*K[2] + KS[1]*K[3]
    KSKt[2] = KS[2]*K[2] + KS[3]*K[3]

    return KSKt


def get_uQ(args, image_size):

    delta_t = args[0]
    index = args[1]
    b = args[2]

    u = np.zeros(shape=(tuple([2])+image_size))
    u[0] = b*delta_t**index
    u[1] = b*(index)*delta_t**(index-1)

    Q = np.zeros(shape=(tuple([3]) +image_size))
    Q[0, :] = delta_t ** (2 * index)
    Q[1, :] = b * delta_t ** (2 * b - 1)
    Q[2, :] = (b ** 2) * (delta_t ** (2 * (b - 1)))
    return u, Q




######## FUNCTIONS #########

def simple_linear(X, args):
    delta_t = args[0]
    flux = X[0, :]+ delta_t*X[1, :]

    rate_flux = X[1, :]

    return np.array([flux, rate_flux])


def identity(X, args):
    return X


def non_linear(X, args):
    delta_t = args[0]
    index = args[1]
    b = args[2]
    #f2 = args[3]

    flux = X[0, :] + b*(delta_t**(index))
    rate_flux = X[1, :] + b*index*(delta_t**(index-1))


    return np.array([flux, rate_flux])

##### MAIN ############


if __name__ == '__main__':
    Xs= sigma_points(mean_=np.ones((2, 4094, 2046))*0.67, cov_= np.ones((3, 4094, 2046)))
    Wm, Wc = unscent_weights()
    #y_mean, y_cov = propagate_func(simple_linear, Wm, Wc, Xs)