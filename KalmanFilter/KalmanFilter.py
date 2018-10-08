from abc import ABC, abstractclassmethod


class KalmanFilter(ABC):

    def __init__(self, image_size):
        self.image_size = image_size

    def correct(self, z, R, pred_state, pred_cov, state, state_cov):
        return self.icorrect.correct(z, R, pred_state,  pred_cov, state, state_cov)

    def predict(self, delta_t, state, state_cov, pred_state, pred_cov):
        return self.ipredict.predict(delta_t, state, state_cov, pred_state, pred_cov)

    def update(self, delta_t, z, R, state, state_cov, pred_state, pred_cov):
        pred_state, pred_cov = self.predict(delta_t, state, state_cov, pred_state, pred_cov)
        return self.correct(z, R, pred_state, pred_cov, state, state_cov)
