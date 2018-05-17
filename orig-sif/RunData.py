import numpy as np
from glob import glob
import sys

from SNDetector import SNDetector
from KalmanFilter import KalmanFilter
from MaximumCorrentropyKalmanFilter import MaximumCorrentropyKalmanFilter

class RunData(object):

    def __init__(self, year='15',
                 only_HiTS_SN=True,
                 test_SN=92,
                 filter_type='kalman',
                 n_params=0,
                 results_dir='.'):
        """

        :param year:
        :param only_HiTS_SN:
        :param test_SN:
        :param filter_type:
        :param n_params:
        :param results_dir:
        """
        self.year = year
        # only CCDs with SN
        self.only_HiTS_SN = only_HiTS_SN

        # Asking if i am @leftraru
        self.at_leftraru = bool(glob('/home/phuente/'))

        self.results_dir = results_dir

        if only_HiTS_SN:
            n_CCDs = 93
        else:
            # sec number is the number of fields
            if self.year == '15':
                n_CCDs = 62 * 56
            else:
                n_CCDs = 62 * 40

        if self.at_leftraru:
            self.index = int(sys.argv[1])
        else:
            # specific SN (test_SN)
            print test_SN
            self.index = test_SN
            #n_params = 0

        self.n_params = n_params
        if self.n_params > 0:
            self.this_par = self.index / n_CCDs
            self.index = self.index % n_CCDs

        self.SN_table = np.loadtxt('./orig-sif/ResultsTable20' + self.year + '.csv', dtype='str', delimiter=',')

        self.images_size = (4094, 2046)

        if self.only_HiTS_SN:
            self.SN_index = self.index
            print self.SN_index
            self.SN_pos = self.SN_table[self.SN_index, [5, 6]].astype(int)
            self.field = self.SN_table[self.SN_index, 3]
            self.ccd = self.SN_table[self.SN_index, 4]
            print 'holiiiii %s' %  self.ccd
            self.resultccd = self.ccd[0] + self.ccd[1:].zfill(2)
        else:
            self.SN_index = -1

        self.filter_type = filter_type

    def apply_params(self):
        """

        :return:
        """
        # 4D grid (extract params)
        decomposing_parameter = self.this_par
        self.filter_type = ['kalman', 'MCC'][decomposing_parameter % 2]
        decomposing_parameter = decomposing_parameter / 2
        # Change threshold
        self.flux_thres = [250, 375, 500, 625][decomposing_parameter % 4]
        decomposing_parameter = decomposing_parameter / 4
        self.vel_flux_thres = [0, 75, 150, 225][decomposing_parameter % 4]

    def deploy_filter_and_detector(self, MJD):
        """

        :param MJD:
        :return:
        """
        self.MJD = MJD
        if self.filter_type == 'kalman':
            KF = KalmanFilter(init_time=self.MJD[0] - 1.0)
        elif self.filter_type == 'MCC':
            KF = MaximumCorrentropyKalmanFilter(init_time=self.MJD[0] - 1.0)
        SN = SNDetector(flux_thres=self.flux_thres, vel_flux_thres=self.vel_flux_thres)
        return KF, SN

    def save_results(self, OB, results_dir='results'):
        """

        :param OB:
        :param results_dir:
        :return:
        """
        filename = self.field + '-' + self.resultccd + '_NUO-' + str(self.NUO).zfill(2)
        if self.SN_index >= 0:
            filename = 'HiTS' + str(self.SN_index + 1).zfill(2) + '-' + ['nay', 'AYE'][self.SN_found] + '_' + filename
        if self.n_params > 0:
            filename = 'par-' + str(self.this_par).zfill(2) + '_' + filename
        np.savez(self.results_dir + '/' + filename, objects=OB.obj)

    def decide_second_run(self, OB):
        """

        :param OB:
        :return:
        """
        #number of unknown object (NUO)
        if self.NUO == 0:
            self.save_results(OB)
            sys.exit(0)
        else:
            SN_data = {}
            SN_data['coords'] = self.SN_pos
            SN_data['epochs'] = []
            SN_data['status'] = self.SN_index + 1
            self.CandData += [SN_data]

