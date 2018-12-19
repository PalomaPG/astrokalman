from .ICorrect import ICorrect

from modules.unscented_utils import *


class UnscentCorrect(ICorrect):

    def __init__(self, Xs, f_func, h_func, Wm=None, Wc=None, alpha=10 ** (-3), beta=2, kappa=0, d=2):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.d = d
        self.h_func = h_func
        self.Xs = Xs
        self.Wm = Wm
        self.Wc = Wc
        self.f_func = f_func

    def correct(self, z, R, pred_state, pred_cov, state, state_cov):
        Ys, Wm, Wc = sigma_points(pred_state, pred_cov, kappa=self.kappa, alpha=self.alpha, beta=self.beta, D=self.d)
        state, state_cov = propagate_func(self.f_func, Wm, Wc, Xs) # z^, S_k - R
        #Residual
        residual = z - state
        S = state_cov + R

        #Innovation

        h_diff = []
        f_diff = []

        for i in range(2*self.d+1):
            h_diff.append(perform(self.h_func, Ys[i]))
            f_diff.append(perform(self.f_func, Xs[i]))



        #Optimal gain
        #K = CS
        state = pred_state# + K*residual
        state_cov = pred_cov# - K*S*K
        return state, state_cov
