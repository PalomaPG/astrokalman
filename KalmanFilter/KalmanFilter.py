from abc import ABC, abstractclassmethod


class KalmanFilter(ABC):

    def __init__(self, image_size):
        self.image_size = image_size

    def correct(self, z, R, pred_state,  pred_cov):
        return self.icorrect.correct(z, R, pred_state,  pred_cov)

    def predict(self, prev_time, curr_time, state, state_cov, pred_state, pred_cov):
        return self.ipredict.predict(prev_time, curr_time, state, state_cov, pred_state, pred_cov)

    @abstractclassmethod
    def update(self, prev_time, curr_time, state, state_cov, pred_state, pred_cov):
        pass
