import unittest
import pandas as pd
import sys
import numpy as np

from sif.FITSHandler import FITSHandler
from sif.RunData import RunData
from sif.KalmanFilter import KalmanFilter
from sif.MaximumCorrentropyKalmanFilter import MaximumCorrentropyKalmanFilter
from sif.SNDetector import SNDetector

from modules.DataPicker import DataPicker
from modules.utils import *
from modules.SourceFinder import SourceFinder
from KalmanFilter.BasicKalman import BasicKalman
from KalmanFilter.MCKalman import MCKalman

class Tests(unittest.TestCase):

    def setUp(self):
        RD = RunData(year='15', n_params=0)
        self.FH = FITSHandler(RD)
        self.MJD = self.FH.MJD
        obs_index_path = '/home/paloma/Documents/Memoria/Code/sif2/inputs/test.csv'
        sn_index = 13
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
        #for i in range(min(len(self.diff_), len(self.FH.data_names['diff']))):
        #    print(self.diff_[i], self.FH.data_names['diff'][i])

    '''
    def test_input(self):

        print('Testing inputs ... Counting elements')
        self.maxDiff = None
        self.assertCountEqual(self.mjd, self.FH.MJD)
        print(len(self.mjd))
        print(len(self.diff_))
        print(len(self.FH.data_names['diff']))
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
        self.FH.load_fluxes(o)
        self.assertListEqual(self.mjd.tolist(), self.FH.MJD.tolist())
        self.assertListEqual(self.sci_, self.FH.data_names['science'])


        np.testing.assert_array_equal(flux, self.FH.flux)
        np.testing.assert_array_equal(var_flux, self.FH.var_flux)


        image_size = (4094, 2046)
        init_state = 0.0
        init_var = 100.0
        state = init_state*np.ones(tuple([2]) + image_size)
        state_cov = np.zeros(tuple([3]) + image_size, dtype=float)
        state_cov[[0, 2], :] = init_var
        pred_state = state.copy()
        pred_cov = state_cov.copy()

        bkf = BasicKalman()
        kf = KalmanFilter()

        np.testing.assert_array_equal(pred_cov, kf.pred_state_cov)
        np.testing.assert_array_equal(pred_state, kf.pred_state)

        np.testing.assert_array_equal(state_cov, kf.state_cov)
        np.testing.assert_array_equal(state, kf.state)


        #kf.update(0.0, self.FH)
        delta_t = self.mjd[o] - 0.0

        
        np.testing.assert_array_equal(pred_cov, kf.pred_state_cov)
        np.testing.assert_array_equal(pred_state, kf.pred_state)
        np.testing.assert_array_equal(state_cov, kf.state_cov)
        np.testing.assert_array_equal(state, kf.state)

        kf.predict_at_new_time(self.mjd[o])
        pred_state, pred_cov = bkf.predict(delta_t, state, state_cov, pred_state, pred_cov)

        np.testing.assert_array_equal(pred_cov, kf.pred_state_cov)
        np.testing.assert_array_equal(pred_state, kf.pred_state)
        np.testing.assert_array_equal(state_cov, kf.state_cov)
        np.testing.assert_array_equal(state, kf.state)

        kf.correct_with_measurements(flux, var_flux)
        state, state_cov = bkf.correct(flux, var_flux, pred_state, pred_cov, state, state_cov)

        np.testing.assert_array_equal(pred_cov, kf.pred_state_cov)
        np.testing.assert_array_equal(pred_state, kf.pred_state)
        np.testing.assert_array_equal(state_cov, kf.state_cov)
        np.testing.assert_array_equal(state, kf.state)

        
        o = 1

        flux, var_flux = calc_fluxes(self.diff_[o], self.psf_[o], self.invvar_[o], self.aflux_[o])
        self.FH.load_fluxes(o)
        np.testing.assert_array_equal(flux, self.FH.flux)
        np.testing.assert_array_equal(var_flux, self.FH.var_flux)

        kf.time = self.mjd[o-1]
        kf.update(self.mjd[o], self.FH)
        state, state_cov = bkf.update((self.mjd[o]-self.mjd[o-1]), flux, var_flux, state, state_cov,
                                      pred_state, pred_cov)

        np.testing.assert_array_equal(state, kf.state)
        np.testing.assert_array_equal(state_cov, kf.state_cov)


    
    def test_MCKF(self):
        print('Filtering with MC Kalman')
        o = 0
        flux, var_flux = calc_fluxes(self.diff_[o], self.psf_[o], self.invvar_[o], self.aflux_[o])


        image_size = (4094, 2046)
        init_state = 0.0
        init_var = 100.0
        state = init_state*np.ones(tuple([2]) + image_size)
        state_cov = np.zeros(tuple([3]) + image_size, dtype=float)
        state_cov[[0, 2], :] = init_var
        pred_state = state.copy()
        pred_cov = state_cov.copy()

        mckf = MCKalman()
        kf = MaximumCorrentropyKalmanFilter()

        np.testing.assert_array_equal(pred_cov, kf.pred_state_cov)
        np.testing.assert_array_equal(pred_state, kf.pred_state)

        np.testing.assert_array_equal(state_cov, kf.state_cov)
        np.testing.assert_array_equal(state, kf.state)

        self.FH.load_fluxes(o)

        #kf.update(0.0, self.FH)
        delta_t = self.mjd[o] - 0.0

        np.testing.assert_array_equal(pred_cov, kf.pred_state_cov)
        np.testing.assert_array_equal(pred_state, kf.pred_state)
        np.testing.assert_array_equal(state_cov, kf.state_cov)
        np.testing.assert_array_equal(state, kf.state)

        kf.predict_at_new_time(self.mjd[o])
        pred_state, pred_cov = mckf.predict(delta_t, state, state_cov, pred_state, pred_cov)

        np.testing.assert_array_equal(pred_cov, kf.pred_state_cov)
        np.testing.assert_array_equal(pred_state, kf.pred_state)
        np.testing.assert_array_equal(state_cov, kf.state_cov)
        np.testing.assert_array_equal(state, kf.state)

        kf.correct_with_measurements(flux, var_flux)
        state, state_cov = mckf.correct(flux, var_flux, pred_state, pred_cov, state, state_cov)

        np.testing.assert_array_equal(pred_cov, kf.pred_state_cov)
        np.testing.assert_array_equal(pred_state, kf.pred_state)
        np.testing.assert_array_equal(state_cov, kf.state_cov)
        np.testing.assert_array_equal(state, kf.state)
     
    '''
    def test_global_basicKF(self):

        image_size = (4094, 2046)
        init_state = 0.0
        init_var = 100.0
        state = init_state*np.ones(tuple([2]) + image_size)
        state_cov = np.zeros(tuple([3]) + image_size, dtype=float)
        state_cov[[0, 2], :] = init_var
        pred_state = state.copy()
        pred_cov = state_cov.copy()

        bkf = BasicKalman()
        kf = KalmanFilter()

        snd = SNDetector()
        sfr = SourceFinder(flux_thresh=500.0, flux_rate_thresh=150.0, rate_satu=3000.0)
        delta_t = (self.mjd[0] - 0.0)
        print(delta_t)

        mask, dil_mask = mask_and_dilation(self.mask_[0])

        for o in range(len(self.mjd)):
            self.FH.load_fluxes(o)
            flux, var_flux = calc_fluxes(self.diff_[o], self.psf_[o], self.invvar_[o], self.aflux_[o])
            np.testing.assert_array_equal(flux, self.FH.flux)
            np.testing.assert_array_equal(var_flux, self.FH.var_flux)

            if o>0:
                delta_t = (self.mjd[o] - self.mjd[o - 1])

            kf.update(self.mjd[o], self.FH)
            print('My delta... %f' % delta_t)
            state, state_cov = bkf.update(delta_t, flux, var_flux, state, state_cov,
                                          pred_state, pred_cov)


            print(self.mjd[o])
            np.testing.assert_array_equal(state, kf.state)
            np.testing.assert_array_equal(state_cov, kf.state_cov)

            sci = fits.open(self.sci_[o])
            snd.pixel_discrimination(o, self.FH, kf)
            sfr.median_rejection, sfr.accum_median_flux = median_rejection_calc(sfr.median_rejection,
                                                                                sfr.accum_median_flux,
                                                                                sfr.accum_med_flux_depth,flux, o)
            pixel_flags = sfr.pixel_discard(sci[0].data, state, state_cov, dil_mask, sfr.median_rejection)

            np.testing.assert_array_equal(sfr.median_rejection, self.FH.median_rejection)
            np.testing.assert_array_equal(pixel_flags, snd.pixel_flags)


            snd.neighboring_pixels()
            sfr.grouping_pixels(pixel_flags, o)
            np.testing.assert_array_equal(sfr.data_content.pixel_mid_coords, snd.PGData['mid_coords'])
            #self.assertListEqual(snd.PGData['pixel_coords'], sfr.data_content.pixel_coords)

            snd.filter_groups(self.FH, kf)
            sfr.filter_groups(sci[0].data, flux, var_flux, state, mask, sfr.median_rejection)
            np.testing.assert_array_equal(snd.PGData['group_flags'], sfr.data_content.group_flags)
            np.testing.assert_array_equal(sfr.data_content.group_flags_map, snd.PGData['group_flags_map'])
            sci.close()


    '''
    def test_global_MCCKF(self):

        image_size = (4094, 2046)
        init_state = 0.0
        init_var = 100.0
        state = init_state*np.ones(tuple([2]) + image_size)
        state_cov = np.zeros(tuple([3]) + image_size, dtype=float)
        state_cov[[0, 2], :] = init_var
        pred_state = state.copy()
        pred_cov = state_cov.copy()

        bkf = MCKalman()
        kf = MaximumCorrentropyKalmanFilter()
        delta_t = (self.mjd[0] - 0.0)
        print(delta_t)

        for o in range(len(self.mjd)):
            self.FH.load_fluxes(o)
            flux, var_flux = calc_fluxes(self.diff_[o], self.psf_[o], self.invvar_[o], self.aflux_[o])
            np.testing.assert_array_equal(flux, self.FH.flux)
            np.testing.assert_array_equal(var_flux, self.FH.var_flux)

            if o>0:
                delta_t = (self.mjd[o] - self.mjd[o - 1])

            kf.update(self.mjd[o], self.FH)
            print('My delta... %f' % delta_t)
            state, state_cov = bkf.update(delta_t, flux, var_flux, state, state_cov,
                                          pred_state, pred_cov)

            print('Testing global basic KF... %d' % o)
            print(self.mjd[o])
            np.testing.assert_array_equal(state, kf.state)
            np.testing.assert_array_equal(state_cov, kf.state_cov)


        
    def test_sndetector(self):
        #Basic Kalman Filter
        #MCC Kalman Filter
        pass
    '''


if __name__ == '__main__':
    unittest.main()
