from .IPredict import IPredict

from modules.unscented_utils import *


class UnscentedPredict(IPredict):

    def __init__(self, f_func, f_args,  Wm, Wc, lambda_, d, sigma_a, image_size= (4094, 2046)):
        self.lambda_ = lambda_
        self.f_func = f_func
        self.d = d
        self.Wc = Wc
        self.Wm = Wm
        self.image_size = image_size
        self.f_args = f_args
        self.sigma_a = sigma_a

    def predict(self, delta_t, state, state_cov, pred_state, pred_cov):
        Xs = sigma_points(state, state_cov, lambda_=self.lambda_, N=self.d)
        Q =get_Q(args=[delta_t]+self.f_args, image_size=self.image_size)
        pred_state, pred_cov = propagate_func_pred(self.f_func, self.Wm, self.Wc, Xs,  (self.sigma_a**2)*Q, delta_t,
                                              args = self.f_args, image_size=self.image_size)
        return pred_state, pred_cov, Xs
