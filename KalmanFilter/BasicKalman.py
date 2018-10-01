import numpy as np
from .KalmanFilter import KalmanFilter
from .LinearPredict import LinearPredict
from .BasicCorrect import BasicCorrect

class BasicKalman(KalmanFilter):

    def __init__(self, z, R, state, state_cov, image_size = (4094, 2046)):
        KalmanFilter.__init__(self, image_size)
        self.ipredict = LinearPredict(state, state_cov)
        self.icorrect = BasicCorrect(z, R, state_cov)

    def update(self, prev_time, curr_time):
        pred_state, pred_cov = self.predict(prev_time, curr_time)
        return self.correct(pred_state, pred_cov)




'''
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
'''