from .ICorrect import ICorrect
from numpy.linalg import inv
from modules.unscented_utils import *


class UnscentCorrect(ICorrect):

    def __init__(self, f_func, h_func, Wm, Wc, lambda_,  d=2):

        self.d = d
        self.lambda_ = lambda_
        self.h_func = h_func
        self.f_func = f_func
        self.Wm = Wm
        self.Wc = Wc

    def define_params(self, *args):
        if len(args) == 1:
            self.Xs = args[0]

    def correct(self, z, R, pred_state, pred_cov, state, state_cov):
        print(self.Xs)
        Ys = sigma_points(pred_state, pred_cov, lambda_=self.lambda_, d=self.d)
        state, state_cov = propagate_func(self.f_func, self.Wm, self.Wc, Ys) # z^, S_k - R
        #Residual
        residual = z - state
        S = state_cov + R

        #Innovation

        h_diff = []
        f_diff = []

        D = 2*self.d+1

        for i in range(D):
            h_diff.append(perform(self.h_func, Ys[i]))
            f_diff.append(perform(self.f_func, Xs[i]))

        for i in range(D):
            h_diff[i] = h_diff[i] - state
            f_diff[i] = f_diff[i] - pred_state

        h_diff = np.array(h_diff)
        f_diff = np.array(f_diff)

        C = [(f_diff[i] @ h_diff[i].T) for i in range(D)]
        C = [self.Wc[i]*C[i] for i in range(D)]
        C = np.sum(C, axis=0)
        #Optimal gain
        K = C @ inv(S)
        state = pred_state + K @ residual
        state_cov = pred_cov - K @ S @ K.T
        return state, state_cov
