from modules.DataPicker import DataPicker
from KalmanFilter.BasicKalman import BasicKalman
from KalmanFilter.MCKalman import MCKalman
from KalmanFilter.UnscentedKalman import UnscentedKalman
from modules.SourceFinder import SourceFinder
from modules.utils import *
from modules.unscented_utils import *
from modules.Observer import Observer

from resource import getrusage as resource_usage, RUSAGE_SELF

import pandas as pd
from os import path, makedirs
import numpy as np
import os
import sys

class RoutineHandler(object):

    def __init__(self, obs_index_path, route_templates, settings_file, index):
        self.obs = (pd.read_csv(obs_index_path, sep=',', header=0, dtype=str, na_values='-')).fillna(-1.0)
        self.route_templates = route_templates
        self.settings = settings_file
        self.index = int(index)

    def process_settings(self):
        self.dict_settings = {}
        with open(self.settings) as f:
            for line in f:
                (key, val) = line.split(sep='=')
                self.dict_settings[key] =  float(val[:-1]) if val[:-1].replace('.', '', 1).isdigit()  else val[:-1]
        self.image_size = (int(self.dict_settings['imgHeight']), int(self.dict_settings['imgWidth']))
        self.dict_settings['UKF'] = False
        self.kf = self.retrieve_kalman_filter(self.dict_settings['filter'])
        self.config_path('plots')
        self.finder = SourceFinder(flux_thresh=self.dict_settings['flux_thresh'],
                              flux_rate_thresh=self.dict_settings['flux_rate_thresh'],
                              rate_satu=self.dict_settings['rate_satu'], image_size= self.image_size)
        self.picker = DataPicker(self.route_templates,self.obs.ix[self.index, 'Semester'], self.obs.ix[self.index, 'Field'],
                            self.obs.ix[self.index, 'CCD'])


    def retrieve_kalman_filter(self, kalman_string):
        if kalman_string == 'mcc':
            print('MCC')
            return MCKalman(sigma_a=self.dict_settings['sigma_a'], std_factor=self.dict_settings['std_factor'],
                            sigma=self.dict_settings['sigma'], max_iter=int(self.dict_settings['max_iter']),
                            epsilon=float(self.dict_settings['epsilon']),
                            silverman_sigma=int(self.dict_settings['silverman']), image_size=self.image_size)
        elif kalman_string=='basic':
            print('Basic')
            return BasicKalman(sigma_a=self.dict_settings['sigma_a'], image_size=self.image_size)
        else:
            print('\nUnscented..............\n')
            self.dict_settings['UKF'] = True

            return UnscentedKalman(non_linear, identity, f_args=[1.0, 1.0],  h_args=[], alpha=self.dict_settings['alpha'],
                                 beta=self.dict_settings['beta'], kappa=self.dict_settings['beta'],
                                 sigma_a=self.dict_settings['sigma_a'], image_size=self.image_size)

    def iterate_over_sequences(self, check_found_objects = False):

        #Setting filenames
        self.diff_ = self.picker.data['diffDir']
        self.psf_ = self.picker.data['psfDir']
        self.invvar_ = self.picker.data['invDir']
        self.aflux_ = self.picker.data['afluxDir']
        self.routine(self.obs.ix[self.index, 'Semester'], self.obs.ix[self.index, 'Field'],
                     self.obs.ix[self.index, 'CCD'], check_found_objects=check_found_objects)

    def config_path(self, output='results'):
        results_path = self.dict_settings[output]
        if not path.exists(results_path):
            makedirs(results_path, exist_ok=True)
        return results_path

    def look_candata(self, observer, data, flux, var_flux, science_, mask, dil_mask, psf, diff, o):
        observer.look_cand_data(data['cand_data'], pred_state=self.kf.pred_state, pred_state_cov=self.kf.pred_cov,
                                kalman_gain=self.kf.kalman_gain, state=self.kf.state, state_cov=self.kf.state_cov,
                                time_mjd=self.picker.mjd[o], flux=flux, var_flux=var_flux, science=science_[0].data,
                                diff=diff, psf=psf, base_mask=mask, dil_base_mask=dil_mask, o=o,
                                group_flags_map=self.finder.PGData['group_flags_map'],
                                pixel_flags=self.finder.pixel_flags)

    def routine_loop(self, init_mjd_idx, n_obs,observer, mask, dil_mask, mjd_o, data, delta_t,
                     check_found_objects = False):
        for o in range(init_mjd_idx, n_obs):

            print('         %d. MJD :   %.2f' % (o, self.picker.mjd[o]))
            print('------------------------------------- \n')

            flux, var_flux, diff, psf = calc_fluxes(self.diff_[o], self.psf_[o], self.invvar_[o], self.aflux_[o])


            if o>0 and not self.dict_settings['UKF']:
                delta_t = self.picker.mjd[o] - self.picker.mjd[o - 1]
            elif o>0 and self.dict_settings['UKF']:
                delta_t = self.picker.mjd[o] - self.picker.mjd[mjd_o]

            self.kf.update(delta_t, flux, var_flux, self.kf.state, self.kf.state_cov, self.kf.pred_state,
                           self.kf.pred_cov)

            science_ = fits.open(self.picker.data['scienceDir'][o])

            self.finder.draw_complying_pixel_groups(science_[0].data, self.kf.state, self.kf.state_cov, mask, dil_mask,
                                               flux, var_flux, o=o)


            if check_found_objects and not self.dict_settings['UKF']:
                self.look_candata(observer, data, flux, var_flux, science_, mask, dil_mask, psf, diff, o)

            elif check_found_objects and self.dict_settings['UKF']:
                self.look_candata(observer, data, flux, var_flux, science_, mask, dil_mask, psf, diff, o-1)

            science_.close()


    def routine(self, semester, field, ccd,  check_found_objects = False, last_mjd=0.0):

        print('-----------------------------------------------------------')
        print('Semester: %s | Field: %s | CCD: %s' % (semester, field, ccd))
        print('-----------------------------------------------------------')

        results_path = self.config_path()
        observer = None
        data = None

        self.kf.define_params(self.dict_settings['init_var'])

        delta_t = self.picker.mjd[0]-last_mjd

        n_obs = len(self.picker.mjd)

        #Mask bad pixels and the neighbors
        mask, dil_mask = mask_and_dilation(self.picker.data['maskDir'][0])
        init_mjd_idx = 0
        stack_length  = n_obs

        if self.dict_settings['UKF']:
            init_mjd_idx = 1
            flux, var_flux, diff, psf = calc_fluxes(self.diff_[0], self.psf_[0], self.invvar_[0], self.aflux_[0])
            self.kf.state[0] = flux
            self.kf.state_cov[0] = var_flux
            stack_length = n_obs-1

        if check_found_objects:
            observer = Observer(stack_length)
            path_npz = ( '%ssources_sem_%s_field_%s_ccd_%s.npz' %
                         (self.dict_settings['cand_data_npz'], semester, field, ccd))
            data = np.load(path_npz)
            if len(data['cand_data']) == 0:
                print('No hay candidatos')
                sys.exit(0)
            observer.set_space(data['cand_data'])

        mjd_o = 0

        self.routine_loop(init_mjd_idx, n_obs, observer, mask, dil_mask, mjd_o, data, delta_t,check_found_objects)

        if not check_found_objects:

            self.finder.check_candidates(self.index, SN_pos=np.array([float(self.obs.ix[self.index, 'POS_Y']),
                                                                 float(self.obs.ix[self.index, 'POS_X'])]))
            output = os.path.join(results_path, 'sources_sem_%s_field_%s_ccd_%s' % (semester, field, ccd))
            np.savez(output, cand_data=self.finder.CandData)

        elif len(observer.obj) > 0:
            plot_path = self.config_path(output='plots')
            observer.plot_results(observer.obj, semester=semester, ccd=ccd, field=field, plot_path=plot_path)



if __name__ == '__main__':
    rh = RoutineHandler(sys.argv[1], sys.argv[2], sys.argv[3])
    rh.process_settings()
    rh.iterate_over_sequences()
    rh.routine('15A', '34', 'N3')
