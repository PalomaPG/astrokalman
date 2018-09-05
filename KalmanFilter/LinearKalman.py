import numpy as np
from .AbstractKalman import AbstractKalman


class LinearKalman(AbstractKalman):

    def __init__(self, init_var=100.0,
                 sigma=1000.0, std_factor=100.0, image_size = (4094, 2046)):
        self.sigma = sigma
        self.std_factor = std_factor
        AbstractKalman.__init__(self, image_size)
        self.state_cov[[0, 2], :] = init_var

    def predict(self, previous_time, current_time, sigma_a=0.1):
        delta_t = current_time - previous_time

        pred_state = np.zeros(self.state.shape)
        pred_state[0] = self.state[0] + self.state[1] * delta_t
        pred_state[1] = self.state[1]

        pred_cov = np.zeros(self.state_cov.shape)
        Q= np.array([delta_t ** 4 / 4, delta_t ** 3 / 2, delta_t ** 2]) * (sigma_a ** 2)
        alpha = self.state_cov[1, :] + delta_t * self.state_cov[2, :] # reserve in original code
        pred_cov[0, :] = self.state_cov[0, :] + delta_t * (self.state_cov[1, :] + alpha) + Q[0]
        pred_cov[1, :] = alpha + Q[1]
        pred_cov[2, :] = self.pred_cov[2, :] + Q[2]

        return  pred_state, pred_cov



    def correct(self, pred_state, pred_cov):
        pass
