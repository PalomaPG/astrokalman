import numpy as np
from .KalmanFilter import KalmanFilter
from .LinearPredict import LinearPredict
from .BasicCorrect import BasicCorrect

class BasicKalman(KalmanFilter):

    def __init__(self, z, R, image_size = (4094, 2046)):
        KalmanFilter.__init__(self, image_size)
        self.ipredict = LinearPredict()
        self.icorrect = BasicCorrect()

    def update(self, prev_time, curr_time, state, state_cov, pred_cov, pred_state=None):
        pred_state, pred_cov = self.predict(prev_time, curr_time, state, state_cov, pred_cov, pred_state=None)
        return self.correct(pred_state, pred_cov)