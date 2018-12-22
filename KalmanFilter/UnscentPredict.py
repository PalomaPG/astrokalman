from .IPredict import IPredict

from modules.unscented_utils import *


class UnscentPredict(IPredict):

    def __init__(self, f_func, Wm, Wc, lambda_, d):
        self.lambda_ = lambda_
        self.f_func = f_func
        self.d = d
        self.Wc = Wc
        self.Wm = Wm

    def predict(self, delta_t, state, state_cov, pred_state, pred_cov):
        Xs = sigma_points(state, state_cov, lambda_=self.lambda_, d=self.d)
        pred_state, pred_cov = propagate_func(self.f_func, self.Wm, self.Wc, Xs, delta_t)
        return pred_state, pred_cov, Xs
