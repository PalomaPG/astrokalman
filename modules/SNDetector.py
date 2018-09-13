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
        self.flux_thres = flux_thres
        self.vel_flux_thres = vel_flux_thres
        self.vel_satu = vel_satu
        self.pixel_conditions = np.zeros(tuple([self.n_conditions]) + images_size, dtype=bool)
        self.pixel_flags = np.zeros(images_size, dtype=int)

    def pixel_discrimination(self, o, science, state):

        science_median = self.subsampled_median(science, 20)
        self.pixel_conditions[:] = False
        self.pixel_flags[:] = 0
        self.pixel_conditions[0, :] = state[0, :] > self.flux_thres
        self.count_pixel_cond_flux.append(sum(sum(self.pixel_conditions[0, :])))

    def check_candidates(self):
        pass

    def update_candidates(self):
        pass

    def draw_complying_pixel_groups(self):
        pass