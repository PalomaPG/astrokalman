import unittest
import pandas as pd
import sys
import numpy as np

from sif.FITSHandler import FITSHandler
from sif.RunData import RunData
from sif.KalmanFilter import KalmanFilter
from sif.MaximumCorrentropyKalmanFilter import MaximumCorrentropyKalmanFilter


from modules.DataPicker import DataPicker
from modules.utils import *
from KalmanFilter.BasicKalman import BasicKalman
from KalmanFilter.MCCKalman import MCCKalman

class Tests(unittest.TestCase):

    def setUp(self):
        RD = RunData(year='15', n_params=0)
        self.FH = FITSHandler(RD)
        self.MJD = self.FH.MJD
        obs_index_path = '/home/paloma/Documents/Memoria/Code/sif2/test.csv'
        sn_index = 13
        config_path = '/home/paloma/Documents/Memoria/Code/sif2/modules/input_example.txt'

        obs = pd.read_csv(obs_index_path, sep=',', header=0)
        obs_info = obs.iloc[[sn_index]]
        picker = DataPicker(config_path, obs_info.iloc[0]['Semester'], obs_info.iloc[0]['Field'],
                            obs_info.iloc[0]['CCD'])
        self.mjd  = np.array(picker.mjd)

        self.diff_ = picker.data['diffDir']
        self.psf_ = picker.data['psfDir']
        self.invvar_ = picker.data['invDir']
        self.aflux_ = picker.data['afluxDir']



    def test_input(self):
        print('Testing inputs ... Counting elements')
        self.assertCountEqual(self.mjd, self.FH.MJD)
        self.assertCountEqual(self.diff_, self.FH.data_names['diff'])
        self.assertCountEqual(self.psf_, self.FH.data_names['psf'])
        self.assertCountEqual(self.invvar_, self.FH.data_names['invVAR'])
        self.assertCountEqual(self.aflux_, self.FH.data_names['aflux'])

        print('Testing inputs ... Counting elements')
        self.assertListEqual(self.mjd.tolist(), self.FH.MJD.tolist())
        self.assertListEqual(self.diff_, self.FH.data_names['diff'])
        self.assertListEqual(self.psf_, self.FH.data_names['psf'])
        self.assertListEqual(self.invvar_, self.FH.data_names['invVAR'])
        self.assertListEqual(self.aflux_, self.FH.data_names['aflux'])


    def test_flux(self):
        print('Calculating and comparing flux info')
        o = 10
        self.FH.load_fluxes(o)
        flux, var_flux = calc_fluxes(self.diff_[o], self.psf_[o], self.invvar_[o], self.aflux_[o])
        np.testing.assert_array_equal(flux, self.FH.flux)
        np.testing.assert_array_equal(var_flux, self.FH.var_flux)

    def test_basicKF(self):
        print('Filtering with Basic Kalman')
        o = 0
        flux, var_flux = calc_fluxes(self.diff_[o], self.psf_[o], self.invvar_[o], self.aflux_[o])
        image_size = (4094, 2046)
        state = np.zeros(tuple([2]) + image_size)
        state_cov = np.zeros(tuple([3]) + image_size, dtype=int)
        bkf = BasicKalman(flux, var_flux, state, state_cov)
        state, state_cov = bkf.update(0.0, self.mjd[0])
        kf = KalmanFilter()
        self.FH.load_fluxes(o)
        kf.update(self.mjd[0], self.FH)
        print(state)
        print(kf.state)
        np.testing.assert_array_equal(kf.state, state)

        #np.testing.assert_array_equal(kf.state_cov, state_cov)



    def test_MCKF(self):
        pass




if __name__ == '__main__':
    unittest.main()
