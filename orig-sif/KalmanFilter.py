# -*- coding: utf-8 -*-

# SIF: Stream Images Filtering

import numpy as np


class KalmanFilter(object):

    def __init__(self, init_time=0.0, init_state=0.0, images_size=(4094, 2046), initial_variance=100.0, sigma=1000.0,
                 std_factor=100.0):
        self.num_states = 2
        self.num_cov_elements = self.num_states * (self.num_states + 1) / 2

        self.time = init_time

        self.state = init_state * np.ones(tuple([self.num_states]) + images_size)

        self.state_cov = np.zeros(tuple([self.num_cov_elements]) + images_size)
        self.state_cov[[0, 2], :] = initial_variance

        self.pred_state = self.state.copy()
        self.pred_state_cov = self.state_cov.copy()

        self.sigma = sigma
        self.std_factor = std_factor
        self.observation = 0

    def variable_accel_Q(self, delta_t, sigma_a=0.1):
        return np.array([delta_t ** 4 / 4, delta_t ** 3 / 2, delta_t ** 2]) * (sigma_a ** 2)

    def predict_at_new_time(self, new_time):
        # Obtain delta_t
        delta_t = new_time - self.time

        # Predict mean
        self.pred_state[0, :] = self.state[0, :] + self.state[1, :] * delta_t
        self.pred_state[1, :] = self.state[1, :].copy()

        # Predict covariance
        Q = self.variable_accel_Q(delta_t)
        reserve = self.state_cov[1, :] + delta_t * self.state_cov[2, :]
        self.pred_state_cov[0, :] = self.state_cov[0, :] + delta_t * (self.state_cov[1, :] + reserve) + Q[0]
        self.pred_state_cov[1, :] = reserve + Q[1]
        self.pred_state_cov[2, :] = self.pred_state_cov[2, :] + Q[2]

    def correct_with_measurements(self, z, R):
        # Obtain inverse of residual's covariance
        inv_S = pow(self.pred_state_cov[0, :] + R, -1)

        # Obtain Kalman Gain
        self.kalman_gain = self.pred_state_cov[[0, 1], :] * inv_S

        # Correct estimate mean
        self.state = self.pred_state + self.kalman_gain * (z - self.pred_state[0, :])

        # Correct estimated covariance (Optimal gain version)
        self.state_cov[[0, 1], :] = self.pred_state_cov[[0, 1], :] * (1.0 - self.kalman_gain[0, :])
        self.state_cov[2, :] = self.pred_state_cov[2, :] - self.kalman_gain[1, :] * self.pred_state_cov[1, :]

    def update(self, new_time, FH):
        # Prediction
        self.predict_at_new_time(new_time)

        # Correction
        self.correct_with_measurements(FH.flux, FH.var_flux)

        # Update time of estimations
        self.time = new_time

        # Update observation index
        self.observation += 1