from .KalmanFilter import KalmanFilter
from .UnscentPredict import UnscentPredict
from .UnscentCorrect import UnscentCorrect


class UnscentKalman(KalmanFilter):

    def __init__(self, f_func, h_func, alpha=10**(-3), beta=2, kappa=0, d=2):
        self.h_func = h_func
        self.ipredict = UnscentPredict(f_func, alpha, beta, kappa, d)
        self.icorrect = UnscentCorrect(h_func, alpha, beta, kappa, d)








