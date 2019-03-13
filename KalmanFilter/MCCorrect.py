from modules.utils import cholesky, image_stats

import numpy as np
from .ICorrect import ICorrect

class MCCorrect(ICorrect):

    def __init__(self, std_factor, sigma, epsilon=1e-6, max_iter=10, silverman_sigma=False):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.silverman_sigma = silverman_sigma
        self.std_factor = std_factor
        self.sigma = sigma

    def correct(self, z, R, pred_state, pred_cov, state, state_cov):
        chol_p, inv_chol_p = cholesky(pred_cov)
        prev_iter_state = pred_state.copy()
        j = 1

        while True:
            C = np.concatenate((pred_state, np.expand_dims(z, 0))) - prev_iter_state[[0, 1, 0], :]
            # Multiply with Cholesky composites
            C[2, :] *= np.power(R, -0.5)  # Inverse of Cholesky decomp of scalar
            C[1, :] = inv_chol_p[1, :] * C[0, :] + inv_chol_p[2, :] * C[1, :]
            C[0, :] *= inv_chol_p[0, :]

            if self.silverman_sigma:
                sigmas = np.zeros(3)
                _, std_pred_flux, _ = image_stats(C[0, :])
                _, std_measurement, _ = image_stats(C[2, :])
                sigmas[2] = self.std_factor * std_measurement
                if np.isnan(std_pred_flux):
                    sigmas[0] = self.std_factor * std_measurement
                else:
                    sigmas[0] = self.std_factor * std_pred_flux
                sigmas[1] = sigmas[0]
                for i in range(C.shape[0]):
                    C[i, :] = np.exp(C[i, :] ** 2 / (2*(sigmas[i] ** 2)))
            else:
                C = np.exp(C ** 2/ (2*(self.sigma ** 2)))

            # Obtain iterative P
            iter_P = chol_p.copy()
            iter_P[[0, 1], :] *= (chol_p[0, :] * C[0, :])

            # Obtain iterative Kalman Gain
            kalman_gain = iter_P[[0, 1], :] / (iter_P[0, :] + C[2, :] * R)

            # Update iterative mean
            iter_state = pred_state + kalman_gain * (z - pred_state[0, :])
            stopped_pixels = np.linalg.norm(iter_state - prev_iter_state, axis=0) <= \
                             np.linalg.norm(prev_iter_state, axis=0) * self.epsilon

            if stopped_pixels.all():
                break
            else:
                j += 1
                prev_iter_state = iter_state.copy()
            if j > self.max_iter:
                break

        # Correct estimated mean
        state = iter_state.copy()

        # Correct estimated covariance
        state_cov[0, :] = np.power(1 - kalman_gain[0, :], 2) * pred_cov[0, :] + np.power(kalman_gain[0, :], 2) * R
        state_cov[1, :] = (1 - kalman_gain[0, :]) * \
                          (pred_cov[1, :] - kalman_gain[1, :] * pred_cov[0, :]) +\
                          kalman_gain[0, :] * kalman_gain[1, :] * R
        state_cov[2, :] = np.power(kalman_gain[1, :], 2) * pred_cov[0, :] - 2.0 * kalman_gain[1, :] * pred_cov[1, :] + \
                          pred_cov[2, :] + np.power(kalman_gain[1, :], 2) * R

        return state, state_cov, kalman_gain