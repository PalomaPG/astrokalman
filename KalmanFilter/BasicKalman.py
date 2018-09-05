import numpy as np
from .LinearKalman import LinearKalman

class BasicKalman(LinearKalman):

    def __init__(self, z, R):
        """
        BasicKalman object initializer
        :param z: flux measurement
        :param R: flux variation from measurement
        """
        self.z = z
        self.R = R
        LinearKalman.__init__()


    def correct(self, pred_state, pred_cov):

        inv_S = pow(pred_cov[0, :] + self.R, -1)
        # Obtain Kalman Gain
        kalman_gain = pred_cov[[0, 1], :] * inv_S
        self.state = pred_state + kalman_gain * (self.z - pred_state[0, :])
        self.state_cov[[0, 1], :] = pred_cov[[0, 1], :] * (1.0 - kalman_gain[0, :])
        self.state_cov[2, :] = pred_cov[2, :] - kalman_gain[1, :] * pred_cov[1, :]