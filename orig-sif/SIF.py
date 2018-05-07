# -*- coding: utf-8 -*-

# SIF: Stream Images Filtering

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pyfits as fits
import scipy.ndimage as spn
import pymorph as pm
import sys


class RunData(object):

    def __init__(self, year='15', only_HiTS_SN=True, test_SN=92, filter_type='kalman', n_params=0,
                 results_dir='results'):
        self.year = year
        # only CCDs with SN
        self.only_HiTS_SN = only_HiTS_SN

        # Asking if i am @leftraru
        self.at_leftraru = bool(glob('/home/phuente/'))

        self.results_dir = results_dir

        if only_HiTS_SN:
            n_CCDs = 93
        else:
            # sec number is the number of fields
            if self.year == '15':
                n_CCDs = 62 * 56
            else:
                n_CCDs = 62 * 40

        if self.at_leftraru:
            self.index = int(sys.argv[1])
        else:
            # specific SN (test_SN)
            self.index = test_SN
            n_params = 0

        self.n_params = n_params
        if self.n_params > 0:
            self.this_par = self.index / n_CCDs
            self.index = self.index % n_CCDs

        self.SN_table = np.loadtxt('./orig-sif/ResultsTable20' + self.year + '.csv', dtype='str', delimiter=',')

        self.images_size = (4096, 2048)

        if self.only_HiTS_SN:
            self.SN_index = self.index
            self.SN_pos = self.SN_table[self.SN_index, [5, 6]].astype(int)
            self.field = self.SN_table[self.SN_index, 3]
            self.ccd = self.SN_table[self.SN_index, 4]
            self.resultccd = self.ccd[0] + self.ccd[1:].zfill(2)
        else:
            self.SN_index = -1

        self.filter_type = filter_type

    def apply_params(self):
        # 4D grid (extract params)
        decomposing_parameter = self.this_par
        self.filter_type = ['kalman', 'MCC'][decomposing_parameter % 2]
        decomposing_parameter = decomposing_parameter / 2
        # Change threshold
        self.flux_thres = [250, 375, 500, 625][decomposing_parameter % 4]
        decomposing_parameter = decomposing_parameter / 4
        self.vel_flux_thres = [0, 75, 150, 225][decomposing_parameter % 4]

    def deploy_filter_and_detector(self, MJD):
        self.MJD = MJD
        if self.filter_type == 'kalman':
            KF = KalmanFilter(init_time=self.MJD[0] - 1.0)
        elif self.filter_type == 'MCC':
            KF = MaximumCorrentropyKalmanFilter(init_time=self.MJD[0] - 1.0)
        SN = SNDetector(flux_thres=self.flux_thres, vel_flux_thres=self.vel_flux_thres)
        return KF, SN

    def save_results(self, OB, results_dir='results'):
        filename = self.field + '-' + self.resultccd + '_NUO-' + str(self.NUO).zfill(2)
        if self.SN_index >= 0:
            filename = 'HiTS' + str(self.SN_index + 1).zfill(2) + '-' + ['nay', 'AYE'][self.SN_found] + '_' + filename
        if self.n_params > 0:
            filename = 'par-' + str(self.this_par).zfill(2) + '_' + filename
        np.savez(self.results_dir + '/' + filename, objects=OB.obj)

    def decide_second_run(self, OB):
        #number of unknown object (NUO)
        if self.NUO == 0:
            self.save_results(OB)
            sys.exit(0)
        else:
            SN_data = {}
            SN_data['coords'] = self.SN_pos
            SN_data['epochs'] = []
            SN_data['status'] = self.SN_index + 1
            self.CandData += [SN_data]


class FITSHandler(object):

    def __init__(self, RD, accum_neg_flux_depth=4, accum_med_flux_depth=3):
        self.field = RD.field
        self.ccd = RD.ccd
        self.year = RD.year
        self.SN_index = RD.SN_index

        self.get_data_names()

        # Classifier criteria aspects
        self.accum_neg_flux_depth = accum_neg_flux_depth
        self.accum_neg_flux = np.zeros(tuple([self.accum_neg_flux_depth]) + RD.images_size, dtype=bool)

        self.accum_med_flux_depth = accum_med_flux_depth
        self.accum_median_flux = np.zeros(tuple([self.accum_med_flux_depth]) + RD.images_size)
        self.median_rejection = np.zeros(RD.images_size, dtype=bool)

    def get_data_names(self):

        self.data_names = {}
        base_dir = '/home/apps/astro/data/ARCHIVE/'

        if glob('/home/phuente/MCKF'):  # At Leftraru
            print 'At Leftraru'

            self.data_names['base'] = \
            sorted(glob(base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/' + self.ccd + '/Blind*_image.fits*'))[
                2]
            self.data_names['base_crblaster'] = sorted(glob(
                base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/' + self.ccd + '/Blind*image_crblaster.fits*'))[
                2]
            # projection
            self.data_names['science'] = sorted(glob(
                base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/' + self.ccd + '/Blind*image_crblaster_grid02_lanczos2.fits'))

            self.data_names['diff'] = []
            self.data_names['invVAR'] = []
            self.data_names['psf'] = []
            self.data_names['aflux'] = []

            for science_filename in self.data_names['science']:
                ind = science_filename.find('_image_')
                epoch = science_filename[ind - 2:ind]
                # difference image
                self.data_names['diff'] += [np.sort(glob(
                    base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/' + self.ccd + '/Diff*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[
                                                0]]
                # 1/var pf diff image
                self.data_names['invVAR'] += [np.sort(glob(
                    base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/' + self.ccd + '/invVAR*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[
                                                  0]]

                # diff image psf
                self.data_names['psf'] += [np.sort(glob(
                    '/home/apps/astro/data/SHARED/Blind' + self.year + 'A_' + self.field + '/' + self.ccd + '/CALIBRATIONS/psf*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[
                                               0]]

                # astrometric and relative flux constants
                self.data_names['aflux'] += [np.sort(glob(
                    '/home/apps/astro/data/SHARED/Blind' + self.year + 'A_' + self.field + '/' + self.ccd + '/CALIBRATIONS/match_*' + epoch + '-02.npy')).tolist()[
                                                 0]]
        else:

            print 'At CMM'

            # baseDir = '/run/media/tesla/Almacen/Huentelemu/R20' + year + 'CCDs/HiTS' + str(snIndex).zfill(2) + 'SN/'
            baseDir = '/home/paloma/Documents/Memoria/data/Blind15A_01/57070.1072727/N1/' # + str(self.SN_index + 1).zfill(2) + 'SN/'
            # baseDir = 'C:/Users/Bahamut/Desktop/HiTS' + str(self.SN_index+1).zfill(2) + 'SN/'
            # baseDir = 'D:/Lab Int Comp/R2015CCDs/HiTS' + str(self.SN_index+1).zfill(2) + 'SN/'

            self.data_names['base'] = [baseDir+'Blind15A_01_N1_57070.1072727_image.fits.fz']
            #glob(baseDir + '/Blind*_02_image.fits*')[0]
            self.data_names['base_crblaster'] = [baseDir+'Blind15A_01_N1_57070.1072727_image_crblaster.fits']
            self.data_names['science'] = np.sort(glob(baseDir + 'science/*')).tolist()

            self.data_names['diff'] = []
            self.data_names['psf'] = []
            self.data_names['invVAR'] = []
            self.data_names['aflux'] = []

            for science_filename in self.data_names['science']:
                ind = science_filename.find('_image_')
                epoch = science_filename[ind - 2:ind]
                self.data_names['diff'] += [
                    np.sort(glob(baseDir + 'diff/Diff*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[0]]
                self.data_names['psf'] += [
                    np.sort(glob(baseDir + 'psf/psf*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[0]]
                self.data_names['invVAR'] += [
                    np.sort(glob(baseDir + 'invvar/invVAR*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[0]]
                self.data_names['aflux'] += [np.sort(glob(baseDir + 'aflux/match_*' + epoch + '-02.npy')).tolist()[0]]

        # Prepare base image

        print self.data_names['base']
        print self.data_names['diff']

        self.base_image = (fits.open(self.data_names['base'])[0]).data
        self.base_mask = fits.open(self.data_names['base'])[1].data

        '''
        dil base mask es una imagen de 0s y 1s?. No se si es la version nueva de python 2.7, o del modulo dilate que no
        pesca el input
        '''
        self.dil_base_mask = pm.dilate(self.base_mask > 0, B=np.ones((5, 5), dtype=bool))

        MJD = [float(fits.open(m)[0].header['MJD-OBS']) for m in self.data_names['science']]
        # Order by MJD
        MJDOrder = np.argsort(MJD)
        MJD = np.array([MJD[i] for i in MJDOrder])

        # Filter airmass
        airmass = np.array([float(fits.open(m)[0].header['AIRMASS']) for m in self.data_names['science']])
        MJD = MJD[airmass < 1.7]
        MJDOrder = MJDOrder[airmass < 1.7]

        for e in ['science', 'diff', 'invVAR', 'psf', 'aflux']:
            if self.data_names.has_key(e):
                self.data_names[e] = [self.data_names[e][i] for i in MJDOrder]

        self.data_names['original_numFrames'] = len(MJD)
        self.data_names['original_MJD'] = MJD

        self.MJD = MJD

    def naylor_photometry(self, invvar):
        self.flux = spn.convolve(self.diff * invvar, self.psf)
        psf2 = self.psf ** 2
        convo = spn.convolve(invvar, psf2)
        convo[convo == 0] = 0.000001
        self.var_flux = 1 / convo
        self.flux = self.flux * self.var_flux

    def load_fluxes(self, o):

        self.science = fits.open(self.data_names['science'][o])[0].data
        self.diff = fits.open(self.data_names['diff'][o])[0].data
        self.psf = np.load(self.data_names['psf'][o])
        invvar = fits.open(self.data_names['invVAR'][o])[0].data

        # Filter bad invVAR values
        invvar[invvar == np.inf] = 0.01

        self.naylor_photometry(invvar)

        # Aflux Correction
        if (self.data_names['diff'][o].find('02t') > 0):
            aflux = np.load(self.data_names['aflux'][o])
            aflux = aflux[0]
            self.flux = self.flux / aflux
            self.var_flux = self.var_flux / (aflux * aflux)

        self.var_flux = np.sqrt(self.var_flux)

        # Filter nan fluxes
        self.flux[np.isnan(self.flux)] = 0.001

        # Register negative fluxes
        self.accum_neg_flux[o % self.accum_neg_flux_depth, :] = self.flux < 0

        # Accumulate fluxes for high initial median rejection
        if o < self.accum_med_flux_depth:
            self.accum_median_flux[o, :] = self.flux
        elif o == self.accum_med_flux_depth:
            self.median_rejection = np.median(self.accum_median_flux, 0) > 1500.0


class KalmanFilter(object):

    def __init__(self, init_time=0.0, init_state=0.0, images_size=(4096, 2048), initial_variance=100.0, sigma=1000.0,
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


class MaximumCorrentropyKalmanFilter(KalmanFilter):

    def image_stats(self, image, outlier_percentage=2.0):
        vector = np.reshape(image, -1)
        max_range = np.mean(np.abs(np.percentile(vector, [outlier_percentage, 100.0 - outlier_percentage])))
        vector = vector[vector < max_range]
        vector = vector[vector > -max_range]
        return np.mean(vector), np.std(vector), vector

    def inv_gauss(self, domain, sigma):
        return np.exp((domain ** 2) / (2 * sigma ** 2))

    def chol2(self, P):
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
            self.kalman_gain = iter_P[[0, 1], :] / (iter_P[0, :] + C[2, :] * R)

            # Update iterative mean
            iter_state = self.pred_state + self.kalman_gain * (z - self.pred_state[0, :])

            # Check stop conditions
            print 'MCKF correction iteration: ' + str(self.n_iter)
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

    def predict_for_Chen_test(self):

        cos = np.cos(np.pi / 18)
        sin = np.sin(np.pi / 18)
        cos2 = cos ** 2
        sin2 = sin ** 2
        cossin = cos * sin

        # Predict mean
        self.pred_state[0, :] = cos * self.state[0, :] - sin * self.state[1, :]
        self.pred_state[1, :] = sin * self.state[0, :] + cos * self.state[1, :]

        # Predict covariance
        Q = np.array([0.01, 0.0, 0.01])
        self.pred_state_cov[0, :] = cos2 * self.state_cov[0, :] - 2 * cossin * self.state_cov[1,
                                                                               :] + sin2 * self.state_cov[2, :] + Q[0]
        self.pred_state_cov[1, :] = cossin * (self.state_cov[0, :] - self.state_cov[2, :]) + (
                    cos2 - sin2) * self.state_cov[1, :] + Q[1]
        self.pred_state_cov[2, :] = sin2 * self.state_cov[0, :] + 2 * cossin * self.state_cov[1,
                                                                               :] + cos2 * self.state_cov[2, :] + Q[2]

    def update_comparison_test_Chen(self, z, R):

        # Prediction
        self.predict_for_Chen_test()

        # Correction
        self.correct_with_measurements(z, R, max_iter=10000)

    def graph_diffs(self, C, figsize1=12, figsize2=8, row=0, outlier_percentage=2.0):
        if row == 0:
            plt.figure(figsize=(figsize1, figsize2))
        titles = ['Flux Prediction Error', 'Velocity Prediction Error', 'Measurement Error']
        for d in range(3):
            plt.subplot2grid((2, 3), (row, d))
            mean, std, vector = self.image_stats(C[d, :])
            var = std ** 2
            quant, bins = np.histogram(vector, bins=200)
            bins = bins[1:]
            plt.plot(bins, quant, '.', label='Histogram')
            plt.plot(bins, max(quant) * np.exp(-0.5 * (bins - mean) ** 2 / var), label='Fit gaussian, std: ' + str(std))
            plt.title(titles[d])
            plt.legend(loc=0)
        if row == 1:
            plt.savefig('iterations/Obs-' + str(self.observation).zfill(2) + '_Iter-' + str(self.n_iter).zfill(2),
                        bbox_inches='tight')
            plt.close('all')


class SNDetector(object):

    def __init__(self, n_consecutive_alerts=4, images_size=(4096, 2048), flux_thres=500.0, vel_flux_thres=150.0,
                 vel_satu=3000.0):
        # n_consecutive_alers -> 4 epochs ago
        self.n_conditions = 7
        self.n_consecutive_alerts = n_consecutive_alerts
        self.pixel_conditions = np.zeros(tuple([self.n_conditions]) + images_size, dtype=bool)
        self.pixel_flags = np.zeros(images_size, dtype=int)
        self.accum_compliant_pixels = np.zeros(tuple([self.n_consecutive_alerts]) + images_size, dtype=bool)
        self.CandData = []
        self.flux_thres = flux_thres
        self.vel_flux_thres = vel_flux_thres
        self.vel_satu = vel_satu

    def subsampled_median(self, image, sampling):
        size1 = 4096
        size2 = 2048
        margin = 100
        yAxis = xrange(margin, size1 - margin, sampling)
        xAxis = xrange(margin, size2 - margin, sampling)
        sampled_image = np.zeros((len(yAxis), len(xAxis)))
        x = 0
        for i in yAxis:
            y = 0
            for j in xAxis:
                sampled_image[x, y] = image[i, j]
                y += 1
            x += 1
        return np.median(sampled_image)

    def pixel_discrimination(self, o, FH, KF):

        epoch_science_median = self.subsampled_median(FH.science, 20)

        self.pixel_conditions[:] = False
        self.pixel_flags[:] = 0

        self.pixel_conditions[0, :] = KF.state[0, :] > self.flux_thres
        self.pixel_conditions[1, :] = KF.state[1, :] > self.vel_flux_thres * (
                    self.vel_satu - np.minimum(KF.state[0, :], self.vel_satu)) / self.vel_satu
        self.pixel_conditions[2, :] = FH.science > epoch_science_median + 5
        self.pixel_conditions[3, :] = KF.state_cov[0, :] < 150.0
        self.pixel_conditions[4, :] = KF.state_cov[2, :] < 150.0
        self.pixel_conditions[5, :] = np.logical_not(FH.dil_base_mask)
        self.pixel_conditions[6, :] = np.logical_not(FH.median_rejection)

        for i in range(self.n_conditions):
            self.pixel_flags[np.logical_not(self.pixel_conditions[i, :])] += 2 ** i

        self.accum_compliant_pixels[o % self.n_consecutive_alerts, :] = self.pixel_flags == 0

    def neighboring_pixels(self):

        self.PGData = {}  # Pixel group data
        self.PGData['pixel_coords'] = []

        alert_pixels = np.all(self.accum_compliant_pixels, 0)

        if not np.any(alert_pixels):
            self.PGData['mid_coords'] = np.zeros((0, 2), dtype=int)
            return

        labeled_image = pm.label(alert_pixels, Bc=np.ones((3, 3), dtype=bool))

        LICoords = np.nonzero(labeled_image)
        LIValues = labeled_image[LICoords]
        LICoords = np.array(LICoords).T

        sortedArgs = np.argsort(LIValues)
        LIValues = LIValues[sortedArgs]
        LICoords = LICoords[sortedArgs, :]

        n_neighboring_pixels = LIValues[-1]

        self.PGData['mid_coords'] = np.zeros((n_neighboring_pixels, 2), dtype=int)

        for i in range(n_neighboring_pixels):
            self.PGData['pixel_coords'] += [LICoords[LIValues == i + 1, :]]
            self.PGData['mid_coords'][i, :] = np.round(np.mean(self.PGData['pixel_coords'][i], 0))

    def filter_groups(self, FH, KF):
        n_pixel_groups = self.PGData['mid_coords'].shape[0]

        self.PGData['group_flags'] = np.zeros(n_pixel_groups, dtype=int)
        self.PGData['group_flags_map'] = -np.ones((4096, 2048), dtype=int)

        for i in range(n_pixel_groups):
            posY, posX = self.PGData['mid_coords'][i, :]

            # Discard groups with negative flux around (bad substractions)
            NNFR = 4
            a, b = posY - NNFR, posY + NNFR + 1
            c, d = posX - NNFR, posX + NNFR + 1
            if np.any(FH.accum_neg_flux[:, a:b, c:d]):
                self.PGData['group_flags'][i] += 1

            # Local Maxima Radius in Science Image
            LMSR = 3
            a, b = posY - LMSR + 1, posY + LMSR + 2
            c, d = posX - LMSR + 1, posX + LMSR + 2
            scienceLM = pm.regmax(FH.science[a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
            if not np.any(scienceLM[1:-1, 1:-1]):
                self.PGData['group_flags'][i] += 2

            # Local Maxima Radius in Flux Image
            LMSR = 3
            a, b = posY - LMSR + 1, posY + LMSR + 2
            c, d = posX - LMSR + 1, posX + LMSR + 2
            fluxLM = pm.regmax(FH.flux[a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
            if not np.any(fluxLM[1:-1, 1:-1]):
                self.PGData['group_flags'][i] += 4

            # Local Maxima Radius in Estimated Flux Velocity Image
            LMSR = 3
            a, b = posY - LMSR + 1, posY + LMSR + 2
            c, d = posX - LMSR + 1, posX + LMSR + 2
            velLM = pm.regmax(KF.state[1, a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
            if not np.any(velLM[1:-1, 1:-1]):
                self.PGData['group_flags'][i] += 8

            # Above local science median
            ASMR = 3
            a, b = posY - ASMR, posY + ASMR + 1
            c, d = posX - ASMR, posX + ASMR + 1
            if not (FH.science[posY, posX] > np.median(FH.science[a:b, c:d]) + 15):
                self.PGData['group_flags'][i] += 16

            # Brightest Pixel on stamps (flux and science)
            BPOS = 10
            a, b = posY - BPOS, posY + BPOS + 1
            c, d = posX - BPOS, posX + BPOS + 1
            brightPixels = np.logical_or(FH.flux[a:b, c:d] > 2 * FH.flux[posY, posX],
                                         FH.science[a:b, c:d] > 2 * FH.science[posY, posX])
            if np.any(brightPixels):
                self.PGData['group_flags'][i] += 32

            # Center over mask
            if FH.base_mask[posY, posX] > 0:
                self.PGData['group_flags'][i] += 64

            # Center over median-rejected pixel
            if FH.median_rejection[posY, posX]:
                self.PGData['group_flags'][i] += 128

            # flux variance
            if FH.var_flux[posY, posX] > 250.0:
                self.PGData['group_flags'][i] += 256

            self.PGData['group_flags_map'][self.PGData['pixel_coords'][i][:, 0], self.PGData['pixel_coords'][i][:, 1]] = \
            self.PGData['group_flags'][i]

    def draw_complying_pixel_groups(self, o, FH, KF):

        # Discriminate every pixel by itself
        self.pixel_discrimination(o, FH, KF)

        # Determine groups of neighboring compliant pixels
        self.neighboring_pixels()

        # Filter groups by morphological analysis
        self.filter_groups(FH, KF)

        print '  Pixel Groups: ' + str(self.PGData['mid_coords'].shape[0])
        print '  Filtered Pixel Groups: ' + str(len(np.nonzero(self.PGData['group_flags'] == 0)[0]))

    def update_candidates(self, o):
        cand_mid_coords = self.PGData['mid_coords'][self.PGData['group_flags'] == 0, :]

        for i in range(cand_mid_coords.shape[0]):

            for c in range(len(self.CandData) + 1):
                if c == len(self.CandData):
                    # New candidate
                    new_cand = {}
                    new_cand['coords'] = cand_mid_coords[i, :]
                    new_cand['epochs'] = [o]
                    self.CandData += [new_cand]
                else:
                    # Part of a previous candidate?
                    if (np.sqrt(np.sum((cand_mid_coords[i, :] - self.CandData[c]['coords']) ** 2)) < 4.0):
                        n_epochs = len(self.CandData[c]['epochs'])
                        self.CandData[c]['coords'] = (self.CandData[c]['coords'] * n_epochs + cand_mid_coords[i, :]) / (
                                    n_epochs + 1)
                        self.CandData[c]['epochs'] += [o]
                        break

    def check_candidates(self, RD):

        RD.NUO = 0  # Number of Unknown Objects

        if RD.SN_index >= 0:
            RD.SN_found = False

            for i in range(len(self.CandData)):
                distance = np.sqrt(np.sum((self.CandData[i]['coords'] - RD.SN_pos) ** 2))

                if distance < 4.0:
                    self.CandData[i]['status'] = 0
                    RD.SN_found = True
                else:
                    self.CandData[i]['status'] = -1
                    RD.NUO += 1
        else:
            RD.NUO = len(self.CandData)
            for i in range(len(self.CandData)):
                self.CandData[i]['status'] = -1

        RD.CandData = self.CandData


class Observer(object):

    def __init__(self, num_obs, obs_rad=10, new_pos=[], figsize1=12, figsize2=8):
        self.figsize1 = figsize1
        self.figsize2 = figsize2
        self.num_obs = num_obs
        self.obs_rad = obs_rad
        self.obs_diam = self.obs_rad * 2 + 1
        self.obj = []
        if len(new_pos) >= 2:
            self.new_object(new_pos[0], new_pos[1], status=1000)

    def new_objects_from_CandData(self, CandData):
        for i in range(len(CandData)):
            self.new_object(CandData[i]['coords'][0], CandData[i]['coords'][1], epochs=CandData[i]['epochs'],
                            status=CandData[i]['status'])

    def new_object(self, posY, posX, epochs=[-1], status=-1):
        new_obj = {'posY': posY, 'posX': posX, 'epochs': epochs, 'status': status}
        new_obj['pred_state'] = np.zeros((self.num_obs, 2, self.obs_diam, self.obs_diam))
        new_obj['pred_state_cov'] = np.zeros((self.num_obs, 3, self.obs_diam, self.obs_diam))
        new_obj['kalman_gain'] = np.zeros((self.num_obs, 2, self.obs_diam, self.obs_diam))
        new_obj['state'] = np.zeros((self.num_obs, 2, self.obs_diam, self.obs_diam))
        new_obj['state_cov'] = np.zeros((self.num_obs, 3, self.obs_diam, self.obs_diam))
        new_obj['MJD'] = np.zeros(self.num_obs)
        new_obj['obs_flux'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['obs_var_flux'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['science'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['diff'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['pixel_flags'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['group_flags'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam))
        new_obj['psf'] = np.zeros((self.num_obs, 21, 21))
        new_obj['base_mask'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam), dtype=int)
        new_obj['dil_base_mask'] = np.zeros((self.num_obs, self.obs_diam, self.obs_diam), dtype=bool)
        self.obj += [new_obj]

    def rescue_run_data(self, o, FH, KF, SND):
        for i in range(len(self.obj)):
            a, b = self.obj[i]['posY'] - self.obs_rad, self.obj[i]['posY'] + self.obs_rad + 1
            c, d = self.obj[i]['posX'] - self.obs_rad, self.obj[i]['posX'] + self.obs_rad + 1
            self.obj[i]['pred_state'][o, :] = KF.pred_state[:, a:b, c:d]
            self.obj[i]['pred_state_cov'][o, :] = KF.pred_state_cov[:, a:b, c:d]
            self.obj[i]['kalman_gain'][o, :] = KF.kalman_gain[:, a:b, c:d]
            self.obj[i]['state'][o, :] = KF.state[:, a:b, c:d]
            self.obj[i]['state_cov'][o, :] = KF.state_cov[:, a:b, c:d]
            self.obj[i]['MJD'][o] = KF.time
            self.obj[i]['obs_flux'][o, :] = FH.flux[a:b, c:d]
            self.obj[i]['obs_var_flux'][o, :] = FH.var_flux[a:b, c:d]
            self.obj[i]['science'][o, :] = FH.science[a:b, c:d]
            self.obj[i]['diff'][o, :] = FH.diff[a:b, c:d]
            self.obj[i]['pixel_flags'][o, :] = SND.pixel_flags[a:b, c:d]
            self.obj[i]['group_flags'][o, :] = SND.PGData['group_flags_map'][a:b, c:d]
            self.obj[i]['psf'][o, :] = FH.psf
            self.obj[i]['base_mask'][o, :] = FH.base_mask[a:b, c:d]
            self.obj[i]['dil_base_mask'][o, :] = FH.dil_base_mask[a:b, c:d]

    def print_lightcurve(self, MJD, obj, posY=-1, posX=-1, save_filename='', SN_found=False):

        num_graphs = 4

        if posY == -1:
            posY = self.obs_rad
        if posX == -1:
            posX = self.obs_rad

        # for i in range(len(self.obj)):
        # obj = self.obj[i]
        # MJD = obj['MJD']

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        ax1 = plt.subplot2grid((num_graphs, 1), (0, 0))
        plt.errorbar(MJD + 0.015, obj['state'][:, 0, posY, posX], yerr=obj['state_cov'][:, 0, posY, posX], fmt='b.-',
                     label='Estimated flux')
        plt.errorbar(MJD - 0.015, obj['pred_state'][:, 0, posY, posX], yerr=obj['pred_state_cov'][:, 0, posY, posX],
                     fmt='g.', label='Predicted flux')
        plt.errorbar(MJD, obj['obs_flux'][:, posY, posX], yerr=obj['obs_var_flux'][:, posY, posX], fmt='r.',
                     label='Observed flux')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)
        plt.ylim([min(obj['state'][:, 0, posY, posX]) - 500, max(obj['state'][:, 0, posY, posX]) + 500])
        plt.title('Position: ' + str(obj['posY']) + ',' + str(obj['posX']) + ', status: ' + str(obj['status']))

        plt.subplot2grid((num_graphs, 1), (1, 0), sharex=ax1)
        plt.errorbar(MJD, obj['state'][:, 1, posY, posX], yerr=obj['state_cov'][:, 2, posY, posX], fmt='b.-',
                     label='Estimated flux velocity')
        plt.errorbar(MJD - 0.03, obj['pred_state'][:, 1, posY, posX], yerr=obj['pred_state_cov'][:, 2, posY, posX],
                     fmt='g.', label='Predicted flux velocity')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)

        plt.subplot2grid((num_graphs, 1), (2, 0), sharex=ax1)
        plt.plot(MJD, obj['pixel_flags'][:, posY, posX], '.-', label='Pixel flags')
        plt.plot(MJD, obj['group_flags'][:, posY, posX], '.-', label='Pixel Group flags')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)

        plt.subplot2grid((num_graphs, 1), (3, 0), sharex=ax1)
        plt.plot(MJD - 0.011, obj['pred_state_cov'][:, 0, posY, posX], 'y.', label='Pred Flux Variance')
        plt.plot(MJD - 0.01, obj['state_cov'][:, 0, posY, posX], 'y+', label='Flux Variance')
        plt.plot(MJD - 0.001, obj['pred_state_cov'][:, 1, posY, posX], 'b.', label='Pred Flux-Velo Variance')
        plt.plot(MJD + 0.00, obj['state_cov'][:, 1, posY, posX], 'b+', label='Flux-Velo Variance')
        plt.plot(MJD + 0.009, obj['pred_state_cov'][:, 2, posY, posX], 'g.', label='Pred Velo Variance')
        plt.plot(MJD + 0.01, obj['state_cov'][:, 2, posY, posX], 'g+', label='Velo Variance')
        plt.grid()
        plt.legend(loc=0, fontsize='small')
        plt.xlim(MJD[0] - 1, MJD[-1] + 1)
        plt.ylim([0, 200])

        plt.xlabel('MJD [days]')

        if len(save_filename) > 0:
            plt.savefig(save_filename + '_lightcurves', bbox_inches='tight')
        plt.close(this_fig)

    def stack_stamps(self, stamps, MJD, max_value=10000):
        stack = stamps[0, :]
        prev_time = MJD[0]
        stamps_diam = stamps.shape[1]
        for i in range(1, stamps.shape[0]):
            stack = np.hstack((stack, max_value * np.ones((stamps_diam, 1))))
            if MJD[i] - prev_time > 0.5:
                stack = np.hstack((stack, -max_value * np.ones((stamps_diam, 1))))
                stack = np.hstack((stack, max_value * np.ones((stamps_diam, 1))))
            stack = np.hstack((stack, stamps[i]))
            prev_time = MJD[i]
        return stack

    def print_stamps(self, MJD, obj, save_filename='', SN_found=False):

        num_graphs = 9

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        plt.subplot2grid((num_graphs, 1), (0, 0))
        plt.imshow(self.stack_stamps(obj['science'], MJD), vmin=0, vmax=600, cmap='Greys_r', interpolation='none')
        plt.axis('off')
        plt.title(
            'Science image, position: ' + str(obj['posY']) + ',' + str(obj['posX']) + ', status: ' + str(obj['status']))

        plt.subplot2grid((num_graphs, 1), (1, 0))
        plt.imshow(self.stack_stamps(obj['psf'], MJD), vmin=0, vmax=0.05, cmap='Greys_r', interpolation='none')
        plt.title('PSF')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (2, 0))
        plt.imshow(self.stack_stamps(obj['obs_flux'], MJD), vmin=-200, vmax=3000, cmap='Greys_r', interpolation='none')
        plt.title('Observed flux')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (3, 0))
        plt.imshow(self.stack_stamps(obj['obs_var_flux'], MJD), vmin=0, vmax=300, cmap='Greys_r', interpolation='none')
        plt.title('Observed flux variance')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (4, 0))
        plt.imshow(self.stack_stamps(obj['state'][:, 0, :], MJD), vmin=-200, vmax=3000, cmap='Greys_r',
                   interpolation='none')
        plt.title('Estimated flux')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (5, 0))
        plt.imshow(self.stack_stamps(obj['state'][:, 1, :], MJD), vmin=-500, vmax=500, cmap='Greys_r',
                   interpolation='none')
        plt.title('Estimated flux velocity')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (6, 0))
        plt.imshow(self.stack_stamps(-obj['pixel_flags'], MJD), vmin=-1, vmax=0, cmap='Greys_r', interpolation='none')
        plt.title('Pixel Flags')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (7, 0))
        plt.imshow(self.stack_stamps(obj['group_flags'], MJD), vmin=-1, vmax=1, cmap='Greys_r', interpolation='none')
        plt.title('Group Flags')
        plt.axis('off')

        plt.subplot2grid((num_graphs, 1), (8, 0))
        plt.imshow(self.stack_stamps(obj['base_mask'], MJD), vmin=0, vmax=1, cmap='Greys_r', interpolation='none')
        plt.title('Base mask')
        plt.axis('off')

        fig.tight_layout()

        if len(save_filename) > 0:
            plt.savefig(save_filename + '_stamps', bbox_inches='tight')
        plt.close(this_fig)

    def print_space_states(self, MJD, obj, posY=-1, posX=-1, save_filename='', SN_found=False):

        if posY == -1:
            posY = self.obs_rad
        if posX == -1:
            posX = self.obs_rad

        this_fig = plt.figure(figsize=(self.figsize1, self.figsize2))

        plt.plot(obj['state'][:, 0, posY, posX], obj['state'][:, 1, posY, posX], 'b.-', label='Estimation')
        plt.plot(obj['obs_flux'][:, posY, posX], np.diff(np.concatenate((np.zeros(1), obj['obs_flux'][:, posY, posX]))),
                 'r.-', label='Observation', alpha=0.25)
        plt.grid()
        plt.plot([500, 500, 3000], [1000, 150, 0], 'k-', label='Thresholds')
        plt.legend(loc=0, fontsize='small')
        plt.plot([500, 3000], [150, 0], 'k-')
        plt.xlim(-500, 3000)
        plt.ylim(-500, 1000)
        plt.title('Position: ' + str(obj['posY']) + ',' + str(obj['posX']) + ', status: ' + str(obj['status']))
        plt.xlabel('Flux [ADU]')
        plt.ylabel('Flux Velocity [ADU/day]')

        if len(save_filename) > 0:
            plt.savefig(save_filename + '_space_states', bbox_inches='tight')
        plt.close(this_fig)

    def print_all_space_states(self, fig, MJD, obj, sn, NUO, SN_found, save_filename=''):

        posY = self.obs_rad
        posX = self.obs_rad

        if NUO:
            if sn == 32:
                return
            plot_color = 'r.-'
        elif SN_found:
            plot_color = 'b.-'
        elif sn < 36:
            plot_color = 'g.-'
        else:
            return

        plt.figure(fig)
        plt.plot(obj['state'][:, 0, posY, posX], obj['state'][:, 1, posY, posX], plot_color, label='Estimation',
                 alpha=0.2)

        if len(save_filename) > 0:
            plt.savefig(save_filename + '_space_states', bbox_inches='tight')
        # plt.close('all')

        return
