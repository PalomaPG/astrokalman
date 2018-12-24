from modules.DataPicker import DataPicker
from KalmanFilter.BasicKalman import BasicKalman
from KalmanFilter.MCKalman import MCKalman
from modules.SourceFinder import SourceFinder
from modules.utils import *
from modules.TPDetector import TPDetector
from modules.DataContent import DataContent
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
        #for index, row in self.obs.iterrows():
        self.routine(self.obs.ix[0,'Semester'],self.obs.ix[0,'Field'],self.obs.ix[0, 'CCD'])

    def routine(self, semester, field, ccd,  last_mjd=0.0):

        print('-----------------------------------------------------------')
        print('Semester: %s | Field: %s | CCD: %s' % (semester, field, ccd))
        print('-----------------------------------------------------------')

        results_path = self.dict_settings['results']
        self.kf.define_params(self.dict_settings['init_var'])

        picker = DataPicker(self.route_templates, semester, field, ccd)

        finder = SourceFinder(flux_thresh=self.dict_settings['flux_thresh'],
                              flux_rate_thresh=self.dict_settings['flux_rate_thresh'],
                              rate_satu=self.dict_settings['rate_satu'], image_size= self.image_size)
        #Setting filenames
        diff_ = picker.data['diffDir']
        psf_ = picker.data['psfDir']
        invvar_ = picker.data['invDir']
        aflux_ = picker.data['afluxDir']

        delta_t = picker.mjd[0]-last_mjd
        n_obs = len(picker.mjd)

        #Mask bad pixels and the neighbors
        mask, dil_mask = mask_and_dilation(picker.data['maskDir'][0])

        for o in range(n_obs):
            data_content = DataContent()
            flux, var_flux, diff, psf = calc_fluxes(diff_[o], psf_[o], invvar_[o], aflux_[o])

            if o>0:
                delta_t = picker.mjd[o] - picker.mjd[o - 1]

            self.kf.update(delta_t, flux, var_flux, self.kf.state, self.kf.state_cov, self.kf.pred_state,
                           self.kf.pred_cov)

            science_ = fits.open(picker.data['scienceDir'][o])
            finder.draw_complying_pixel_groups(science_[0].data, self.kf.state, self.kf.state_cov, mask, dil_mask,
                                               flux, var_flux, picker.mjd[o], field, ccd, results_path,
                                               data_content=data_content, o=o)

            data_content.save_results(results_path, field, ccd, semester, science=science_[0].data, obs_flux=flux,
                                      obs_flux_var=var_flux, state=self.kf.state, state_cov=self.kf.state_cov,
                                      diff=diff, psf=psf, mask=mask, dil_mask=dil_mask, mjd=picker.mjd[o],
                                      pred_state=self.kf.pred_state, pred_state_cov=self.kf.pred_cov,
                                      pixel_flags=finder.pixel_flags)
            science_.close()

        print('---------------------------------------------------------------------')

    def get_results(self):
        tpd = TPDetector()
        cands = tpd.look_candidates(self.dict_settings['results'], ccd='N9', field='41')
        print(cands)
        #tpd.get_plots(coords=cands[4], results_path=self.dict_settings['results'], field=41, ccd='N9')

if __name__ == '__main__':
    rh = RoutineHandler(sys.argv[1], sys.argv[2], sys.argv[3])
    rh.process_settings()
    #rh.iterate_over_sequences()
    rh.routine('15A', '34', 'N3')
    #tpd = TPDetector()
    #tpd.look_candidates(results_path, ccd='N27', field='22')