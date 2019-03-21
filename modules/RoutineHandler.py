from modules.DataPicker import DataPicker
from KalmanFilter.BasicKalman import BasicKalman
from KalmanFilter.MCKalman import MCKalman
from KalmanFilter.UnscentKalman import UnscentKalman
from modules.SourceFinder import SourceFinder
from modules.utils import *
from modules.unscented_utils import *
from modules.Observer import Observer


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
        self.kf = self.retrieve_kalman_filter(self.dict_settings['filter'])
        self.config_path('plots')

    def retrieve_kalman_filter(self, kalman_string):
        if kalman_string == 'mcc':
            print('MCC')
            return MCKalman(sigma_a=self.dict_settings['sigma_a'], std_factor=self.dict_settings['std_factor'],
                            sigma=self.dict_settings['sigma'], max_iter=int(self.dict_settings['max_iter']),
                            epsilon=float(self.dict_settings['epsilon']),
                            silverman_sigma=int(self.dict_settings['silverman_sigma']), image_size=self.image_size)
        elif kalman_string=='basic':
            print('Basic')
            return BasicKalman(sigma_a=self.dict_settings['sigma_a'], image_size=self.image_size)
        else:
            print('Unscented')
            return UnscentKalman(non_linear, identity, f_args=[1.5, 1.0],  h_args=[],
                                 sigma_a=self.dict_settings['sigma_a'], image_size=self.image_size)

    def iterate_over_sequences(self, check_found_objects = False):
        self.routine(self.obs.ix[self.index, 'Semester'], self.obs.ix[self.index, 'Field'],
                     self.obs.ix[self.index, 'CCD'], check_found_objects=check_found_objects)

    def config_path(self, output='results'):
        results_path = self.dict_settings[output]
        if not path.exists(results_path):
            makedirs(results_path, exist_ok=True)
        return results_path


    def routine(self, semester, field, ccd,  check_found_objects = False, last_mjd=0.0):

        print('-----------------------------------------------------------')
        print('Semester: %s | Field: %s | CCD: %s' % (semester, field, ccd))
        print('-----------------------------------------------------------')
        #print(np.array([float(self.obs.ix[self.index, 'POS_Y']), float(self.obs.ix[self.index, 'POS_X'])]))

        results_path = self.config_path()

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

        if check_found_objects:
            print('NPZ file...')
            tpd = Observer(n_obs)
            path_npz = ( '%ssources_sem_%s_field_%s_ccd_%s.npz' %
                         (self.dict_settings['cand_data_npz'], semester, field, ccd))
            data = np.load(path_npz)
            tpd.set_space(data['cand_data'])
        #Mask bad pixels and the neighbors
        mask, dil_mask = mask_and_dilation(picker.data['maskDir'][0])

        for o in range(n_obs):

            print('         %d. MJD :   %.2f' % (o, picker.mjd[o]))
            print('------------------------------------- \n')
            flux, var_flux, diff, psf = calc_fluxes(diff_[o], psf_[o], invvar_[o], aflux_[o])

            if o>0:
                delta_t = picker.mjd[o] - picker.mjd[o - 1]


            self.kf.update(delta_t, flux, var_flux, self.kf.state, self.kf.state_cov, self.kf.pred_state,
                           self.kf.pred_cov)

            science_ = fits.open(picker.data['scienceDir'][o])
            finder.draw_complying_pixel_groups(science_[0].data, self.kf.state, self.kf.state_cov, mask, dil_mask,
                                               flux, var_flux, o=o)

            if check_found_objects:
                tpd.look_cand_data(data['cand_data'], pred_state=self.kf.pred_state, pred_state_cov=self.kf.pred_cov,
                                   kalman_gain=self.kf.kalman_gain, state=self.kf.state, state_cov=self.kf.state_cov,
                                   time_mjd=picker.mjd[o],flux=flux, var_flux=var_flux,science=science_[0].data,
                                   diff=diff, psf=psf,base_mask=mask, dil_base_mask=dil_mask, o=o,
                                   group_flags_map=finder.PGData['group_flags_map'],
                                   pixel_flags=finder.pixel_flags)

            science_.close()

        if not check_found_objects:

            finder.check_candidates(self.index, SN_pos=np.array([float(self.obs.ix[self.index, 'POS_Y']),
                                                                 float(self.obs.ix[self.index, 'POS_X'])]))
            output = os.path.join(results_path, 'sources_sem_%s_field_%s_ccd_%s' % (semester, field, ccd))
            np.savez(output, cand_data=finder.CandData)
            print('Number of unidentified objects: ' + str(finder.NUO))
        elif len(tpd.obj) > 0:
            print(tpd.obj[0]['MJD'])
            plot_path = self.config_path(output='plots')
            tpd.plot_results(tpd.obj, semester=semester, ccd=ccd, field=field, plot_path=plot_path)
            #np.savez(path_npz, objects=tpd.obj)

    def plot_results(self):

        semester = self.obs.ix[self.index, 'Semester']
        field = self.obs.ix[self.index, 'Field']
        ccd = self.obs.ix[self.index, 'CCD']
        picker = DataPicker(self.route_templates, semester, field, ccd)

        tpd = Observer(picker.mjd)
        path_npz = ('%ssources_sem_%s_field_%s_ccd_%s.npz' %
                    (self.dict_settings['cand_data_npz'], semester, field, ccd))
        data = np.load(path_npz)


        #tpd.look_candidates(self.dict_settings['results'], ccd=self.obs.ix[self.index, 'CCD'], field=self.obs.ix[self.index, 'Field'])


if __name__ == '__main__':
    rh = RoutineHandler(sys.argv[1], sys.argv[2], sys.argv[3])
    rh.process_settings()
    rh.iterate_over_sequences()
    rh.routine('15A', '34', 'N3')
