import numpy as np
from .LinearKalman import LinearKalman
from modules.utils import cholesky, image_stats


class MCCKalman(LinearKalman):

    def __init__(self, z, R, epsilon=1e-6, max_iter=10, silverman_sigma=False):
        """
        BasicKalman object initializer
        :param z: flux measurement
        :param R: flux variation from measurement
        """
        self.z = z
        self.R = R
        self.epsilon=epsilon
        self.max_iter= max_iter
        self.silverman_sigma = silverman_sigma
        LinearKalman.__init__()

    def correct(self, pred_state, pred_cov):
        L_p, inv_L_p = cholesky(pred_cov)
        prev_iter_state = pred_state.copy()

        for i in range(self.max_iter):
            C = np.concatenate((pred_state, np.expand_dims(self.z, 0))) - prev_iter_state[[0, 1, 0], :]
            C[2, :] *= np.power(self.R, -0.5)
            C[1, :] = inv_L_p[1, :] * C[0, :] + inv_L_p[2, :] * C[1, :]
            C[0, :] *= inv_L_p[0, :]
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
                    C[i, :] = np.exp((C[i, :] ** 2) / (2 * sigmas[i] ** 2))#self.inv_gauss(C[i, :], sigmas[i])
            else:
                C = np.exp((C** 2) / (2 * self.sigma** 2))#self.inv_gauss(C, self.sigma)

            # Obtain iterative P
            iter_P = L_p.copy()
            iter_P[[0, 1], :] *= (L_p[0, :] * C[0, :])
            kalman_gain = iter_P[[0, 1], :] / (iter_P[0, :] + C[2, :] * self.R)
            iter_state = pred_state + kalman_gain * (self.z - pred_state[0, :])
            stopped_pixels = np.linalg.norm(iter_state - prev_iter_state, axis=0) \
                             <= np.linalg.norm(prev_iter_state, axis=0) * self.epsilon
            if stopped_pixels.all():
                break
            prev_iter_state = iter_state.copy()

        # Correct estimated covariance
        self.state_cov[0, :] = np.power(1 - kalman_gain[0, :], 2) * self.pred_cov[0, :] + np.power(
            kalman_gain[0, :], 2) * self.R
        self.state_cov[1, :] = (1 - kalman_gain[0, :]) * \
                               (self.pred_cov[1, :] - kalman_gain[1, :] * self.pred_cov[0, :])\
                               + kalman_gain[0,:] * kalman_gain[1, :] * self.R

        self.state_cov[2, :] = np.power(kalman_gain[1, :], 2) * self.pred_cov[0, :] - \
                               2.0 * kalman_gain[1, :] * self.pred_cov[1, :] +\
                               self.pred_cov[2,:] + np.power( kalman_gain[1, :], 2) * self.R

