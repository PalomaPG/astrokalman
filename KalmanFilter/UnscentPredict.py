from .IPredict import IPredict

from modules.unscented_utils import *


class UnscentPredict(IPredict):

    def __init__(self, f_func, alpha=10**(-3), beta=2, kappa=0, d=2):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.d = d
        self.f_func = f_func

    def predict(self, delta_t, state, state_cov, pred_state, pred_cov):
        Xs, Wm, Wc = sigma_points(state, state_cov, kappa=self.kappa, alpha=self.alpha, beta=self.beta, D=self.d)
        pred_state, pred_cov = propagate_func(self.f_func, Wm, Wc, Xs)
        return pred_state, pred_cov
