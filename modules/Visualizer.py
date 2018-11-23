import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, num_obs, results_path, plots_path,
                 obs_rad=10, new_pos=[], figsize1=12, figsize2=8):

        self.results_path = results_path
        self.plots_path = plots_path

        self.figsize1 = figsize1
        self.figsize2 = figsize2
        self.num_obs = num_obs
        self.obs_rad = obs_rad
        self.obs_diam = self.obs_rad * 2 + 1
        self.obj = []
        if len(new_pos) >= 2:
            self.new_object(new_pos[0], new_pos[1], status=1000)


    def plot_candidate(self, ccd, field, semester):
        pass