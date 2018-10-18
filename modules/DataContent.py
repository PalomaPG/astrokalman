import numpy as np
class DataContent(object):

    def __init__(self, n_neighboring_pixels):
        self.pixel_coords = []
        self.candidates_data = []
        self.pixel_mid_coords = np.zeros((n_neighboring_pixels, 2), dtype=int)

    def group_info(self, image_size):
        n_pixel_groups = self.pixel_mid_coords.shape[0]
        self.group_flags = np.zeros(n_pixel_groups, dtype=int)
        self.group_flags_map = -np.ones(image_size, dtype=int)
        return n_pixel_groups

    def save_data(self):
        pass
