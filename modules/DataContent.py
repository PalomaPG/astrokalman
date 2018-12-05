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

    def save_results(self, path_, field, ccd, semester,  science, obs_flux, obs_flux_var, state, state_cov, pred_state,
                     pred_state_cov, diff, psf, mask, dil_mask, mjd, pixel_flags):
        """

        :param path_:
        :return:
        """
        cand_mid_coords = self.pixel_mid_coords[self.group_flags == 0, :]
        out_temp = os.path.join(path_, 'sources_sem_%s_mjd_%.2f_field_%s_ccd_%s' % (semester, mjd, field, ccd))

        np.savez(out_temp, pixel_coords=self.pixel_coords, pixel_mid_coords=self.pixel_mid_coords,
                 cand_mid_coords=cand_mid_coords,  science=science, obs_flux= obs_flux, obs_flux_var=obs_flux_var,
                 state=state, state_cov=state_cov, diff=diff, psf=psf, mask=mask, dil_mask=dil_mask, mjd=mjd,
                 pred_state=pred_state, pred_state_cov = pred_state_cov, pixel_group_flags = self.group_flags,
                 pixel_flags=pixel_flags)
