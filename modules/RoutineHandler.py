from modules.DataPicker import DataPicker
from KalmanFilter.BasicKalman import BasicKalman
from KalmanFilter.MCKalman import MCKalman
from modules.SourceFinder import SourceFinder
from modules.utils import *
from modules.TPDetector import TPDetector
from resource import getrusage as resource_usage, RUSAGE_SELF

import pandas as pd
import numpy as np
import sys

class RoutineHandler(object):

    def __init__(self, obs_index_path, route_templates, settings_file):
        self.obs = pd.read_csv(obs_index_path, sep=',', header=0)
        self.route_templates = route_templates
        self.settings = settings_file

    def process_settings(self):
        self.dict_settings = {}
        with open(self.settings) as f:
            for line in f:
                (key, val) = line.split(sep='=')
                self.dict_settings[key] =  float(val[:-1]) if val[:-1].replace('.', '', 1).isdigit()  else val[:-1]
        self.image_size = (int(self.dict_settings['imgHeight']), int(self.dict_settings['imgWidth']))
        self.kf = self.retrieve_kalman_filter(self.dict_settings['filter'])

    def retrieve_kalman_filter(self, kalman_string):
        if kalman_string == 'MCC':
            return MCKalman()
        else:
            return BasicKalman()

    def iterate_over_sequences(self):
        for index, row in self.obs.iterrows():
            self.routine(row['Semester'],row['Field'], row['CCD'])

    def routine(self, semester, field, ccd,  last_mjd=0.0):

        print('------------------------------------------------------')
        print('Semester: %s | Field: %s | CCD: %s' % (semester, field, ccd))
        print('------------------------------------------------------')

        results_path = self.dict_settings['results']
        t_i = resource_usage(RUSAGE_SELF).ru_utime
        picker = DataPicker(self.route_templates, semester, field, ccd)
        t_f = resource_usage(RUSAGE_SELF).ru_utime
        print('Data preparation: %f' % (t_f-t_i))

        finder = SourceFinder(flux_thresh=self.dict_settings['flux_thresh'],
                              flux_rate_thresh=self.dict_settings['flux_rate_thresh'],
                              rate_satu=self.dict_settings['rate_satu'], image_size= self.image_size)
        #Setting filenames
        diff_ = picker.data['diffDir']
        psf_ = picker.data['psfDir']
        invvar_ = picker.data['invDir']
        aflux_ = picker.data['afluxDir']

        state = np.zeros(tuple([2]) + self.image_size, dtype=float)
        state_cov = np.zeros(tuple([3]) + self.image_size, dtype=float)
        state_cov[[0, 2], :] = self.dict_settings['init_var']
        pred_state = state.copy()
        pred_cov = state_cov.copy()

        delta_t = picker.mjd[0]-last_mjd
        n_obs = len(picker.mjd)

        mask, dil_mask = mask_and_dilation(picker.data['maskDir'][0])
        calc_flux_time = 0.0
        filter_time = 0.0
        draw_groups_time = 0.0

        for o in range(n_obs):
            t_i = resource_usage(RUSAGE_SELF).ru_utime
            flux, var_flux = calc_fluxes(diff_[o], psf_[o], invvar_[o], aflux_[o])

            t_f = resource_usage(RUSAGE_SELF).ru_utime
            calc_flux_time += (t_f - t_i)
            if o>0:
                delta_t = picker.mjd[o] - picker.mjd[o - 1]

            t_i = resource_usage(RUSAGE_SELF).ru_utime
            state, state_cov = self.kf.update(delta_t, flux, var_flux, state, state_cov,
                                          pred_state, pred_cov)
            t_f = resource_usage(RUSAGE_SELF).ru_utime
            filter_time += (t_f - t_i)



            t_i = resource_usage(RUSAGE_SELF).ru_utime
            science_ = fits.open(picker.data['scienceDir'][o])
            finder.draw_complying_pixel_groups(science_[0].data, state, state_cov, mask, dil_mask,
                                               flux, var_flux, picker.mjd[o], field, ccd, results_path, o=o,
                                               median_reject=finder.median_rejection)
            t_f = resource_usage(RUSAGE_SELF).ru_utime
            draw_groups_time +=(t_f-t_i)


            science_.close()
        #self.look_candidates(results_path, field, ccd)
        print('Tiempo de calculo de flujo: %f' % calc_flux_time)
        print('Tiempo filtro de Kalman: %f' % filter_time )
        print('draw_groups_time: %f' % draw_groups_time)

        tpd = TPDetector()
        tpd.look_candidates(results_path, ccd=ccd, field=field)

        print('---------------------------------------------------------------------')

if __name__ == '__main__':
    rh = RoutineHandler(sys.argv[1], sys.argv[2], sys.argv[3])
    rh.process_settings()
    #rh.iterate_over_sequences()
    rh.routine('15A', '34', 'N3')
    #tpd = TPDetector()
    #tpd.look_candidates(results_path, ccd='N27', field='22')