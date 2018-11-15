import numpy as np
import os
class DataContent(object):

    def __init__(self):
        """

        :param n_neighboring_pixels:
        :param field:
        :param ccd:
        :param mjd:
        """
        self.pixel_coords = []
        self.pixel_mid_coords = np.zeros((0, 2), dtype=int)


    def set_mid_coords(self,  n_neighboring_pixels):
        self.pixel_mid_coords = np.zeros((n_neighboring_pixels, 2), dtype=int)

    def group_info(self, image_size):
        """

        :param image_size:
        :return:
        """
        n_pixel_groups = self.pixel_mid_coords.shape[0]
        self.group_flags = np.zeros(n_pixel_groups, dtype=int)
        self.group_flags_map = -np.ones(image_size, dtype=int)

        return n_pixel_groups

    def save_results(self, path_, field, ccd, mjd, state=None, state_cov=None, save_state_info=False, semester='15A'):
        """

        :param path_:
        :return:
        """
        cand_mid_coords = self.pixel_mid_coords[self.group_flags == 0, :]
        out_temp = os.path.join(path_, 'sources_sem_%s_mjd_%.2f_field_%s_ccd_%s' % (semester, mjd, field, ccd))
        if not save_state_info:
            np.savez(out_temp, pixel_coords=self.pixel_coords, pixel_mid_coords=self.pixel_mid_coords,
                 cand_mid_coords=cand_mid_coords)
        else:
            np.savez(out_temp, pixel_coords=self.pixel_coords, pixel_mid_coords=self.pixel_mid_coords,
                     cand_mid_coords=cand_mid_coords, last_state=state, last_state_cov=state_cov)