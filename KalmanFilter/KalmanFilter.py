from abc import ABC
import numpy as np

class KalmanFilter(ABC):

    def __init__(self, image_size):
        self.image_size = image_size
        self.kalman_gain = 0.0

    def correct(self, z, R, pred_state, pred_cov, state, state_cov, delta_t=0.0):
        return self.icorrect.correct(z, R, pred_state,  pred_cov, state, state_cov, delta_t)

    def predict(self, delta_t, state, state_cov, pred_state, pred_cov):
        return self.ipredict.predict(delta_t, state, state_cov, pred_state, pred_cov)

    def update(self, delta_t, z, R, state, state_cov, pred_state, pred_cov):
        self.pred_state, self.pred_cov = self.predict(delta_t, state, state_cov, pred_state, pred_cov)
        self.state, self.state_cov, self.kalman_gain = self.correct(z, R, self.pred_state, self.pred_cov, state,
                                                                    state_cov, delta_t)

    def define_params(self, init_var):
        self.state = np.zeros(tuple([2]) + self.image_size, dtype=float)
        self.state_cov = np.zeros(tuple([3]) + self.image_size, dtype=float)
        self.state_cov[[0, 2], :] = init_var
        self.pred_state = self.state.copy()
        self.pred_cov = self.state_cov.copy()
