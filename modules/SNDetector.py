import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
from .utils import *


class SNDetector(object):

    def __init__(self, path_inputs):
        pass

    def select_candidates(self, data_content):
        #cand_mid_coords = self.PGData['mid_coords'][self.PGData['group_flags'] == 0, :]
        cand_mid_coords = data_content.pixel_mid_coords[data_content.group_flags == 0, :]
