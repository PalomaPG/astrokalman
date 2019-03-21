from modules.unscented_utils import unscent_weights
from .KalmanFilter import KalmanFilter
from .UnscentPredict import UnscentPredict
from .UnscentCorrect import UnscentCorrect

import numpy as np

class UnscentKalman(KalmanFilter):

    def __init__(self, f_func, h_func, f_args, h_args, alpha=10**(-3), beta=2, kappa=0, d=2, sigma_a=0.1, image_size = (4094, 2046)):
        Wm, Wc = unscent_weights(kappa=kappa, alpha=alpha, N=d, beta=beta)
        lambda_ = (alpha**2)*(d + kappa) - d
        self.ipredict = UnscentPredict(f_func, f_args, Wm, Wc, lambda_, d, sigma_a, image_size)
        self.icorrect = UnscentCorrect(f_func, h_func, f_args, h_args, Wm, Wc, lambda_, d, image_size)
        self.image_size = image_size

    def update(self, delta_t, z, R, state, state_cov, pred_state, pred_cov):
        self.pred_state, self.pred_cov, Xs = self.predict(delta_t, state, state_cov, pred_state, pred_cov)
        self.icorrect.define_params(Xs)
        self.state, self.state_cov,  self.kalman_gain  = self.correct(z, R, self.pred_state, self.pred_cov, state,
                                                                      state_cov, delta_t)







