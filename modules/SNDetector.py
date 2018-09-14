import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
from .utils import *


class SNDetector(object):

    def __init__(self, flux_thres, vel_flux_thres, vel_satu, images_size):
        """

        :param flux_thres:
        :param vel_flux_thres:
        :param vel_satu:
        :param images_size:
        """
        self.n_conditions = 7
        self.flux_thres = flux_thres
        self.vel_flux_thres = vel_flux_thres
        self.vel_satu = vel_satu
        self.pixel_conditions = np.zeros(tuple([self.n_conditions]) + images_size, dtype=bool)
        self.pixel_flags = np.zeros(images_size, dtype=int)
        self.PGData = {}  # Pixel group data


    def pixel_discrimination(self, o, science, state, kf, dil_mask, median_rejection):

        science_median = self.subsampled_median(science, 20)
        self.pixel_conditions[:] = False
        self.pixel_flags[:] = 0
        self.pixel_conditions[0, :] = state[0, :] > self.flux_thres
        self.count_pixel_cond_flux.append(sum(sum(self.pixel_conditions[0, :])))
        self.pixel_conditions[1, :] = kf.state[1, :] > (self.vel_flux_thres * (
                    self.vel_satu - np.minimum(kf.state[0, :], self.vel_satu)) / self.vel_satu)
        self.pixel_conditions[2, :] = science > science_median + 5
        self.pixel_conditions[3, :] = kf.state_cov[0, :] < 150.0 #check value
        self.pixel_conditions[4, :] = kf.state_cov[2, :] < 150.0
        self.pixel_conditions[5, :] = np.logical_not(dil_mask)
        self.pixel_conditions[6, :] = np.logical_not(median_rejection)

        for i in range(self.n_conditions):
            self.pixel_flags[np.logical_not(self.pixel_conditions[i, :])] += 2 ** i

        #self.accum_compliant_pixels[o % self.n_consecutive_alerts, :] = (self.pixel_flags == 0)

    def neighboring_pixels(self):

        self.PGData['pixel_coords'] = []
        alert_pixels = np.all(self.accum_compliant_pixels, 0)
        if not np.any(alert_pixels):
            self.PGData['mid_coords'] = np.zeros((0, 2), dtype=int)
        labeled_image, nr_objects = mh.label(alert_pixels, Bc=np.ones((3, 3), dtype=int))

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