import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
from .utils import *


class SNDetector(object):

    def __init__(self, alerts = 4, flux_thres=500.0, flux_rate_thres=150.0, rate_satu=3000.0, image_size=(4094, 2046)):

        """

        :param alerts: # consecutive alerts
        :param flux_thres: flux threshold
        :param flux_rate_thres: flux rate threshold
        :param rate_satu: flux rate saturation level
        :param images_size: size of science FITS images
        """
        self.alerts = alerts
        self.n_conditions = 7
        self.flux_thres = flux_thres
        self.flux_rate_thres = flux_rate_thres
        self.rate_satu = rate_satu

        # Data structures
        self.accum_compliant_pixels = np.zeros(tuple([alerts]) + image_size, dtype=bool)
        self.pixel_conditions = np.zeros(tuple([self.n_conditions]) + image_size, dtype=bool) # conditions per pixel
        self.pixel_flags = np.zeros(image_size, dtype=int) # flags per pixel
        self.PGData = {}  # Pixel group data


    def pixel_discard(self, o, science, state, state_cov, dil_mask, median_reject):

        """

        :param science: Science image
        :param state: State (image size array) previously obtained by Kalman filter
        :param state_cov: Covariance array determined by kalman filter
        :param dil_mask: dilation mask
        :param median_rejection:
        :return:
        """

        science_median = self.subsampled_median(science, 20)

        self.pixel_conditions[:] = False
        self.pixel_flags[:] = 0


        self.pixel_conditions[0, :] = state[0, :] > self.flux_thres
        self.pixel_conditions[1, :] = state[1, :] > (self.vel_flux_thres * (
                    self.vel_satu - np.minimum(state[0, :], self.vel_satu)) / self.vel_satu)
        self.pixel_conditions[2, :] = science > science_median + 5
        self.pixel_conditions[3, :] = state_cov[0, :] < 150.0 #check value
        self.pixel_conditions[4, :] = state_cov[2, :] < 150.0
        self.pixel_conditions[5, :] = np.logical_not(dil_mask)
        self.pixel_conditions[6, :] = np.logical_not(median_reject)

        for i in range(self.n_conditions):
            self.pixel_flags[np.logical_not(self.pixel_conditions[i, :])] += 2 ** i

        self.accum_compliant_pixels[o % self.alerts, :] = (self.pixel_flags == 0)

    def grouping_pixels(self):

        self.PGData['pixel_coords'] = []
        alert_pixels = np.all(self.accum_compliant_pixels, 0)

        #if not discarding criteria
        if not np.any(alert_pixels):
            self.PGData['mid_coords'] = np.zeros((0, 2), dtype=int)
        # Recognizing closed areas and labeling them
        labeled_image, nr_objects = mh.label(alert_pixels, Bc=np.ones((3, 3), dtype=int))

        LICoords = np.nonzero(labeled_image)
        LIValues = labeled_image[np.nonzero(labeled_image)]
        LICoords = np.array(LICoords).T

        sortedArgs = np.argsort(LIValues)
        LIValues = LIValues[sortedArgs]
        LICoords = LICoords[sortedArgs, :]

        n_neighboring_pixels = LIValues[-1]
        self.PGData['mid_coords'] = np.zeros((n_neighboring_pixels, 2), dtype=int)

        for i in range(n_neighboring_pixels):

            self.PGData['pixel_coords'] += [LICoords[LIValues == i + 1, :]]
            self.PGData['mid_coords'][i, :] = np.round(np.mean(self.PGData['pixel_coords'][i], 0))

    def filter_groups(self):

        n_pixel_groups = self.PGData['mid_coords'].shape[0]
        self.PGData['group_flags'] = np.zeros(n_pixel_groups, dtype=int)
        self.PGData['group_flags_map'] = -np.ones((4094, 2046), dtype=int)
        for i in range(n_pixel_groups):
            posY, posX = self.PGData['mid_coords'][i, :]
            """
            NNFR = 4
            a, b = posY - NNFR, posY + NNFR + 1
            c, d = posX - NNFR, posX + NNFR + 1
            if np.any(FH.accum_neg_flux[:, a:b, c:d]):
                self.PGData['group_flags'][i] += 1
            """

        pass

    def draw_complying_pixel_groups(self):
        pass