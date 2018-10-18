import numpy as np
import os
class DataContent(object):

    def __init__(self, n_neighboring_pixels, field, ccd, mjd):
        self.pixel_coords = []
        self.pixel_mid_coords = np.zeros((n_neighboring_pixels, 2), dtype=int)
        self.field = field
        self.ccd = ccd
        self.mjd = mjd

    def group_info(self, image_size):
        n_pixel_groups = self.pixel_mid_coords.shape[0]
        self.group_flags = np.zeros(n_pixel_groups, dtype=int)
        self.group_flags_map = -np.ones(image_size, dtype=int)

        return n_pixel_groups

    def save_data(self, path_):
        cand_mid_coords = self.pixel_mid_coords[self.group_flags == 0, :]
        out_temp = os.path.join(path_, 'sources_mjd_%.2f_field_%s_ccd_%s.txt' % (self.mjd, self.field, self.ccd))
        np.savez(out_temp, pixel_coords=self.pixel_coords, pixel_mid_coords=self.pixel_mid_coords,
                 cand_mid_coords=cand_mid_coords)
