# -*- coding: utf-8 -*-

# SIF: Stream Images Filtering

import numpy as np

from sif.KalmanFilter import KalmanFilter


class MaximumCorrentropyKalmanFilter(KalmanFilter):


    def image_stats(self, image, outlier_percentage=2.0):
        """

        :param image:
        :param outlier_percentage:
        :return:
        """
        vector = np.reshape(image, -1)
        max_range = np.mean(np.abs(np.percentile(vector, [outlier_percentage, 100.0 - outlier_percentage])))
        vector = vector[vector < max_range]
        vector = vector[vector > -max_range]
        return np.mean(vector), np.std(vector), vector

    def inv_gauss(self, domain, sigma):
        """

        :param domain:
        :param sigma:
        :return:
        """
        return np.exp((domain ** 2) / (2 * sigma ** 2))

    def chol2(self, P):
        """

        :param P:
        :return:
        """
        # Cholesky decomposition
        L = np.zeros(P.shape)
        L[0, :] = np.sqrt(P[0, :])
        L[1, :] = P[1, :] / L[0, :]
        L[2, :] = np.sqrt(P[2, :] - np.power(L[1, :], 2))
        # Include inversion
        inv_L = np.ones(L.shape)
        inv_L[1, :] = -L[1, :]
        inv_L[[0, 1], :] = inv_L[[0, 1], :] / L[0, :]
        inv_L[[1, 2], :] = inv_L[[1, 2], :] / L[2, :]
        return L, inv_L

    def correct_with_measurements(self, z, R, epsilon=1e-6, max_iter=10, silverman_sigma=False, report_graphs=False):
        """

        :param z:
        :param R:
        :param epsilon:
        :param max_iter:
        :param silverman_sigma:
        :param report_graphs:
        :return:
        """
        # Obtain Cholesky decomposition
        L_p, inv_L_p = self.chol2(self.pred_state_cov)

        # Begin fixed point iterations
        prev_iter_state = self.pred_state.copy()
        self.n_iter = 1
        while True:
            # Obtain iterative C

            # Start with states substraction
            C = np.concatenate((self.pred_state, np.expand_dims(z, 0))) - prev_iter_state[[0, 1, 0], :]
            if report_graphs:
                self.graph_diffs(C)

            # Multiply with Cholesky composites
            C[2, :] *= np.power(R, -0.5)  # Inverse of Cholesky decomp of scalar
            C[1, :] = inv_L_p[1, :] * C[0, :] + inv_L_p[2, :] * C[1, :]
            C[0, :] *= inv_L_p[0, :]
            if report_graphs:
                self.graph_diffs(C, row=1)

            # Obtain inversed Gaussians
            if silverman_sigma:
                sigmas = np.zeros(3)
                _, std_pred_flux, _ = self.image_stats(C[0, :])
                _, std_measurement, _ = self.image_stats(C[2, :])
                sigmas[2] = self.std_factor * std_measurement
                if np.isnan(std_pred_flux):
                    sigmas[0] = self.std_factor * std_measurement
                else:
                    sigmas[0] = self.std_factor * std_pred_flux
                sigmas[1] = sigmas[0]
                for i in range(C.shape[0]):
                    C[i, :] = self.inv_gauss(C[i, :], sigmas[i])
            else:
                C = self.inv_gauss(C, self.sigma)
            # Obtain iterative P
            iter_P = L_p.copy()
            iter_P[[0, 1], :] *= (L_p[0, :] * C[0, :])
            # iter_P[2,:] = L_p[1,:]**2*C[0,:] + L_p[2,:]**2*C[1,:] # Unnecessary math

            # Obtain iterative Kalman Gain
            self.kalman_gain = iter_P[[0, 1], :] / (iter_P[0, :]+ C[2, :] * R)

            # Update iterative mean
            iter_state = self.pred_state + self.kalman_gain * (z - self.pred_state[0, :])

            # Check stop conditions
            print('MCKF correction iteration: ' + str(self.n_iter))
            stopped_pixels = np.linalg.norm(iter_state - prev_iter_state, axis=0) <= np.linalg.norm(prev_iter_state,
                                                                                                    axis=0) * epsilon
            # print 'Remaining pixels: ' +str(4096.0*2048.0-len(np.nonzero(stopped_pixels)[0]))
            # print np.max()
            if stopped_pixels.all():
                break
            else:
                self.n_iter += 1
                prev_iter_state = iter_state.copy()
            if self.n_iter > max_iter:
                break

        # Correct estimated mean
        self.state = iter_state.copy()

        # Correct estimated covariance
        self.state_cov[0, :] = np.power(1 - self.kalman_gain[0, :], 2) * self.pred_state_cov[0, :] + np.power(
            self.kalman_gain[0, :], 2) * R
        self.state_cov[1, :] = (1 - self.kalman_gain[0, :]) * (
                    self.pred_state_cov[1, :] - self.kalman_gain[1, :] * self.pred_state_cov[0, :]) + self.kalman_gain[
                                                                                                      0,
                                                                                                      :] * self.kalman_gain[
                                                                                                           1, :] * R
        self.state_cov[2, :] = np.power(self.kalman_gain[1, :], 2) * self.pred_state_cov[0, :] - 2.0 * self.kalman_gain[
                                                                                                       1,
                                                                                                       :] * self.pred_state_cov[
                                                                                                            1,
                                                                                                            :] + self.pred_state_cov[
                                                                                                                 2,
                                                                                                                 :] + np.power(
            self.kalman_gain[1, :], 2) * R
