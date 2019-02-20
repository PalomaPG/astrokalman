import numpy as np
import os
from scipy.spatial import distance as dist

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

    def save_results(self, path_, field, ccd, semester, state, state_cov, pred_state,
                     pred_state_cov, mjd, pixel_flags,  science_name, diff_name, psf_name, aflux_name,
                     mask_name, invvar_name):
        """

        :param path_:
        :return:
        """
        cand_mid_coords = self.pixel_mid_coords[self.group_flags == 0, :]
        output = os.path.join(path_, 'sources_sem_%s_mjd_%.2f_field_%s_ccd_%s' % (semester, mjd, field, ccd))

        cand_mid_coords = self.filter_cand_mid_coords(cand_mid_coords)

        np.savez(output, pixel_coords=self.pixel_coords, pixel_mid_coords=self.pixel_mid_coords,
                 cand_mid_coords=cand_mid_coords,
                 state=state, state_cov=state_cov, mjd=mjd,
                 pred_state=pred_state, pred_state_cov = pred_state_cov, pixel_group_flags = self.group_flags_map,
                 pixel_flags=pixel_flags, science_name = science_name, diff_name = diff_name, psf_name = psf_name,
                 aflux_name = aflux_name, invvar_name = invvar_name, mask_name=mask_name)


    def filter_cand_mid_coords(self, coords):

        aux_coords = []
        for c in coords:
            if len(aux_coords) == 0:
                aux_coords.append(c)
            else:
                isin = False
                for a_c in aux_coords:
                    if dist.euclidean(a_c, c) <= 4:
                        isin = True
                if not isin:
                    aux_coords.append(c)

        return aux_coords
