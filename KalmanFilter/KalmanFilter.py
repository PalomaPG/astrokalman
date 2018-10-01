import numpy as np

from abc import ABC, abstractclassmethod

class KalmanFilter(ABC):

    def __init__(self, image_size):
        self.image_size = image_size

    def correct(self, pred_state,  pred_cov):
        self.icorrect.correct(pred_state,  pred_cov)

    def predict(self, prev_time, curr_time):
        self.ipredict.predict(prev_time, curr_time)

    def update(self, prev_time, curr_time):
        pred_state, pred_cov = self.predict(prev_time, curr_time)
        self.correct(pred_state, pred_cov)