from modules.unscented_utils import unscent_weights
from .KalmanFilter import KalmanFilter
from .UnscentPredict import UnscentPredict
from .UnscentCorrect import UnscentCorrect


class UnscentKalman(KalmanFilter):

    def __init__(self, f_func, h_func, alpha=10**(-3), beta=2, kappa=0, d=2):
        Wm, Wc = unscent_weights(kappa=kappa, alpha=alpha, D=d, beta=beta)
        lambda_ = (alpha**2)*(d + kappa) - d
        self.ipredict = UnscentPredict(f_func, Wm, Wc, lambda_, d)
        self.icorrect = UnscentCorrect(f_func, h_func, Wm, Wc, lambda_, d)

    def update(self, delta_t, z, R, state, state_cov, pred_state, pred_cov):
        pred_state, pred_cov, Xs = self.predict(delta_t, state, state_cov, pred_state, pred_cov)
        return self.correct(z, R, pred_state, pred_cov, state, state_cov)







