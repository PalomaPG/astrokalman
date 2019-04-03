import numpy as np
from .IPredict import IPredict


class LinearPredict(IPredict):

    def __init__(self, sigma_a=0.1):
        self.sigma_a = sigma_a

    def predict(self, delta_t, state, state_cov, pred_state, pred_cov):

        pred_state[0] = state[0] + state[1] * delta_t
        pred_state[1] = state[1]

        Q = np.array([delta_t ** 4 / 4, delta_t ** 3 / 2, delta_t ** 2]) * (self.sigma_a ** 2)
        print('Q...')
        print(np.nanmin(Q[0]))
        print(np.nanmax(Q[1]))

        alpha = state_cov[1, :] + delta_t * state_cov[2, :] # reserve in original code
        print('alpha...')
        print(np.nanmin(alpha))
        print(np.nanmax(alpha))

        pred_cov[0, :] = state_cov[0, :] + delta_t * (state_cov[1, :] + alpha) + Q[0]
        pred_cov[1, :] = alpha + Q[1]
        pred_cov[2, :] = state_cov[2, :] + Q[2]
        print('PREDCOV')
        print(np.nanmax(pred_cov[0]))
        print(np.nanmin(pred_cov[1]))
        print(np.nanmin(pred_cov[2]))

        return pred_state, pred_cov