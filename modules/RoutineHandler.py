from .DataPicker import DataPicker
from KalmanFilter import BasicKalman, MCKalman
from .utils import *
import pandas as pd
import numpy as np
import sys


class RoutineHandler(object):

    def __init__(self, obs_index_path, route_templates):
        self.obs = pd.read_csv(obs_index_path, sep=',', header=0)
        self.route_templates = route_templates

    def method1(self, ccd_field_sem):

        #ccd_field_sem = self.obs.iloc[[ccd_field_index]]
        picker = DataPicker(self.route_templates, ccd_field_sem.iloc[0]['Semester'], ccd_field_sem.iloc[0]['Field'],
                            ccd_field_sem.iloc[0]['CCD'])

        data = picker.get_data()
        self.routine(data, picker.mjd)

        del picker
        del data

    def routine(self, data, mjds):

        for i in range(len(mjds)):
            flux, var_flux = calc_fluxes(data['diffDir'][i], data['psfDir'][i], data['invDir'][i], data['afluxDir'][i])
            bk = BasicKalman()







