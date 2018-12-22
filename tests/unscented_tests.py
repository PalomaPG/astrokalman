import unittest
import pandas as pd
import sys
import numpy as np


from modules.DataPicker import DataPicker
from modules.utils import *
from modules.unscented_utils import simple_linear, identity
from modules.SourceFinder import SourceFinder
from KalmanFilter.UnscentKalman import UnscentKalman
from KalmanFilter.BasicKalman import BasicKalman


class Tests(unittest.TestCase):

    def setUp(self):

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

        pred_state, pred_cov = bkf.predict(delta_t=self.mjd[0], state=state, state_cov=state_cov, pred_state=pred_state,
                                           pred_cov=pred_cov)

        np.testing.assert_array_equal(u_pred_state, pred_state)
        ukf.icorrect.define_params(Xs)

        #np.testing.assert_array_almost_equal(u_pred_cov, pred_cov)

if __name__ == '__main__':
    unittest.main()