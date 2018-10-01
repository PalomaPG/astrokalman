import numpy as np
from .ICorrect import ICorrect

class BasicCorrect(ICorrect):

    def __init__(self,  z, R, state_cov):
        self.z = z
        self.R = R
        self.state_cov = state_cov

    def correct(self, pred_state,  pred_cov):
        inv_S = pow(pred_cov[0, :] + self.R, -1)
        # Obtain Kalman Gain
        kalman_gain = pred_cov[[0, 1], :] * inv_S
        state = pred_state + kalman_gain * (self.z - pred_state[0, :])
        self.state_cov[[0, 1], :] = pred_cov[[0, 1], :] * (1.0 - kalman_gain[0, :])
        self.state_cov[2, :] = pred_cov[2, :] - kalman_gain[1, :] * pred_cov[1, :]

        return state, self.state_cov

    def redefine(self, state_cov):
        self.state_cov = state_cov