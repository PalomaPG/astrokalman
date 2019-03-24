import unittest
import pandas as pd
import sys
import numpy as np


from modules.DataPicker import DataPicker
from modules.utils import *
from modules.unscented_utils import simple_linear, identity, non_linear
from modules.SourceFinder import SourceFinder
from KalmanFilter.UnscentedKalman import UnscentKalman
from KalmanFilter.BasicKalman import BasicKalman


class Tests(unittest.TestCase):

    def setUp(self):
        self.ukf_lin = UnscentKalman(non_linear, non_linear, f_args=[1.0, 1.0], h_args=[1.0, 0.0], image_size=(2,2))
        self.ukf_nonlin = UnscentKalman(non_linear, non_linear, f_args=[2.0, 1.0],  h_args=[1.0, 0.0],  image_size=(2,2))

        '''
        obs_index_path = '/home/paloma/Documents/Memoria/Code/sif2/inputs/test.csv'
        sn_index = 0
        config_path = '/home/paloma/Documents/Memoria/Code/sif2/inputs/input_example.txt'

        obs = pd.read_csv(obs_index_path, sep=',', header=0)
        obs_info = obs.iloc[[sn_index]]
        picker = DataPicker(config_path, obs_info.iloc[0]['Semester'], obs_info.iloc[0]['Field'],
                            obs_info.iloc[0]['CCD'])
        self.mjd  = np.array(picker.mjd)
        self.sci_ = picker.data['scienceDir']
        self.diff_ = picker.data['diffDir']
        self.psf_ = picker.data['psfDir']
        self.invvar_ = picker.data['invDir']
        self.aflux_ = picker.data['afluxDir']
        self.mask_ = picker.data['maskDir']
        '''
    def test_lin(self):
        print('BEGIN LINEAR TEST')
        init_state = 0.0
        init_var = 100.0
        image_size = (2, 2)
        state = init_state * np.ones(tuple([2]) + image_size, dtype=float)
        state_cov = np.zeros(tuple([3]) + image_size, dtype=float)
        state_cov[[0, 2], :] = init_var
        pred_state = state.copy()
        pred_cov = state_cov.copy()
        delta_t = 2
        state[0,:] = 2.0*np.ones(image_size)
        pred_state, pred_cov, Xs = self.ukf_lin.predict(delta_t=delta_t, state=state, state_cov=state_cov,
                                               pred_state=pred_state, pred_cov=pred_cov)
        self.ukf_lin.icorrect.define_params(Xs)
        z = 34.0*np.ones(image_size)
        R = 150.0*np.ones(image_size, dtype=float)
        print('\nPRED STATE\n')
        print(pred_state)
        print(pred_cov)

        state, state_cov, kalman_gain = self.ukf_lin.correct(z, R, pred_state, pred_cov, state, state_cov)
        print('\nSTATE\n')
        print(state)
        print(state_cov)
        print(kalman_gain)
        pred_state, pred_cov, Xs = self.ukf_lin.predict(delta_t=delta_t, state=state, state_cov=state_cov,
                                               pred_state=pred_state, pred_cov=pred_cov)
        print('\nPRED STATE\n')
        print(pred_state)
        print(pred_cov)

        self.ukf_lin.icorrect.define_params(Xs)
        z = 57.0 * np.ones(image_size)
        state, state_cov, kalman_gain = self.ukf_lin.correct(z, R, pred_state, pred_cov, state, state_cov)
        print('\nSTATE\n')

        print(state)
        print(state_cov)
        print(kalman_gain)

        pred_state, pred_cov, Xs = self.ukf_lin.predict(delta_t=delta_t, state=state, state_cov=state_cov,
                                               pred_state=pred_state, pred_cov=pred_cov)
        print('\nPRED STATE\n')
        print(pred_state)
        print(pred_cov)
        self.ukf_lin.icorrect.define_params(Xs)
        z = 85.0 * np.ones(image_size)
        state, state_cov, kalman_gain = self.ukf_lin.correct(z, R, pred_state, pred_cov, state, state_cov)
        print(state)
        print(state_cov)
        print(kalman_gain)

        pred_state, pred_cov, Xs = self.ukf_lin.predict(delta_t=delta_t, state=state, state_cov=state_cov,
                                               pred_state=pred_state, pred_cov=pred_cov)
        print('\nPRED STATE\n')
        print(pred_state)
        print(pred_cov)

        print('END LINEAR TEST')

"""
    def test_nonlin(self):
        init_state = 0.0
        init_var = 100.0
        image_size = (2, 2)
        state = init_state * np.ones(tuple([2]) + image_size, dtype=float)
        state_cov = np.zeros(tuple([3]) + image_size, dtype=float)
        state_cov[[0, 2], :] = init_var
        pred_state = state.copy()
        pred_cov = state_cov.copy()
        delta_t = 2
        state[0,:] = 2.0*np.ones(image_size)
        pred_state_nonlin, pred_cov_nonlin, Xs_nonlin = self.ukf_nonlin.predict(delta_t=delta_t, state=state, state_cov=state_cov,
                                               pred_state=pred_state, pred_cov=pred_cov)

        self.ukf_nonlin.icorrect.define_params(Xs_nonlin)
        z = 34.0*np.ones(image_size)
        R = 15.0*np.ones(image_size, dtype=float)
        print(pred_state_nonlin)
        print(pred_cov_nonlin)
        state_nonlin, state_cov_nonlin, kalman_gain = self.ukf_nonlin.correct(z, R, pred_state_nonlin, pred_cov_nonlin, state, state_cov)

        print(state_nonlin)
        print(state_cov_nonlin)

        pred_state_nonlin, pred_cov_nonlin, Xs_nonlin = self.ukf_nonlin.predict(delta_t=delta_t, state=state_nonlin, state_cov=state_cov_nonlin,
                                               pred_state=pred_state_nonlin, pred_cov=pred_cov_nonlin)
        self.ukf_nonlin.icorrect.define_params(Xs_nonlin)
        z = 53.0*np.ones(image_size)
        state_nonlin, state_cov_nonlin, kalman_gain = self.ukf_nonlin.correct(z, R, pred_state_nonlin, pred_cov_nonlin, state_nonlin, state_cov_nonlin)
        print(state_nonlin)
        print(state_cov_nonlin)
"""


"""
    def test_unscent(self):
        print('Filtering with Unscent Kalman')
        o = 0
        flux, var_flux, diff_data, psf = calc_fluxes(self.diff_[o], self.psf_[o], self.invvar_[o], self.aflux_[o])

        image_size = (4094, 2046)
        init_state = 0.0
        init_var = 100.0

        
        state = init_state*np.ones(tuple([2]) + image_size)
        state_cov = np.zeros(tuple([3]) + image_size, dtype=float)
        state_cov[[0, 2], :] = init_var
        pred_state = state.copy()
        pred_cov = state_cov.copy()


        u_state = init_state*np.ones(tuple([2]) + image_size)
        u_state_cov = np.zeros(tuple([3]) + image_size, dtype=float)
        u_state_cov[[0, 2], :] = init_var
        u_pred_state = u_state.copy()
        u_pred_cov = u_state_cov.copy()

        bkf = BasicKalman()
        ukf = UnscentKalman(simple_linear, identity)
        u_pred_state, u_pred_cov, Xs = ukf.predict(delta_t=self.mjd[0], state=u_state, state_cov=u_state_cov,
                                               pred_state=u_pred_state, pred_cov=u_pred_cov)

        #pred_state, pred_cov = bkf.predict(delta_t=self.mjd[0], state=state, state_cov=state_cov, pred_state=pred_state,
        #                                   pred_cov=pred_cov)

        #np.testing.assert_array_equal(u_pred_state, pred_state)
        ukf.icorrect.define_params(Xs, self.mjd[0], image_size)
        u_state, u_state_cov = ukf.correct(flux, var_flux, pred_cov=u_pred_cov, pred_state=u_pred_state, state=u_state,
                                           state_cov=u_state_cov)
        #np.testing.assert_array_almost_equal(u_pred_cov, pred_cov)
"""

if __name__ == '__main__':
    unittest.main()