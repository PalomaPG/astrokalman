import numpy as np
from glob import glob
import sys
import os

from SNDetector import SNDetector
from KalmanFilter import KalmanFilter
from MaximumCorrentropyKalmanFilter import MaximumCorrentropyKalmanFilter

class RunData(object):

    def __init__(self, year='15',
                 only_HiTS_SN=True,
                 test_SN=92,
                 filter_type='kalman',
                 n_params=0,
                 results_dir='./results/'):
        """
        Guarda parametros de la ejecucion
        :param year: string, sn year (13,14,15)
        :param only_HiTS_SN: bool, solo datos con sn de HiTS
        :param test_SN: int, SN to test (no para leftraru)
        :param filter_type: string, identifica filtro: Kalman, MCKF, etc
        :param n_params: int, # veces que tiene que repetirse el algoritmo (solo leftraru)
        :param results_dir: string, Directorio resultados
        """
        self.year = year
        # only CCDs with SN
        self.only_HiTS_SN = only_HiTS_SN

        # Asking if i am @leftraru
        self.at_leftraru = bool(glob('/home/pperez/'))

        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

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
            #print test_SN
            self.index = test_SN
            #n_params = 0

        self.n_params = n_params
        if self.n_params > 0:
            print(self.index)
            print(n_CCDs)
            self.this_par = int(self.index / n_CCDs)
            print(int(self.this_par))
            self.index = self.index % n_CCDs

        self.SN_table = np.loadtxt('./ResultsTable20' + self.year + '.csv', dtype='str', delimiter=',')

        self.images_size = (4094, 2046)

        if self.only_HiTS_SN:
            self.SN_index = self.index
            #print self.SN_index
            self.SN_pos = self.SN_table[self.SN_index, [5, 6]].astype(int)
            self.field = self.SN_table[self.SN_index, 3]
            self.ccd = self.SN_table[self.SN_index, 4]
            print(self.ccd)
            self.resultccd = self.ccd[0] + self.ccd[1:].zfill(2)
        else:
            self.SN_index = -1

        self.filter_type = filter_type

    def apply_params(self):
        """
        Setea los parametros segun la variable self.this_par
        :return void:
        """
        # 4D grid (extract params)
        decomposing_parameter = self.this_par
        print(self.this_par)
        self.filter_type = ['kalman', 'MCC'][decomposing_parameter % 2]
        #decomposing_parameter = decomposing_parameter / 2
        decomposing_parameter = int(decomposing_parameter / 2)
        # Change threshold
        self.flux_thres = [250, 375, 500, 625][decomposing_parameter % 4]
        decomposing_parameter = int(decomposing_parameter / 4)
        self.vel_flux_thres = [0, 75, 150, 225][decomposing_parameter % 4]

    def deploy_filter_and_detector(self, MJD):
        """
        Prepara filtro y el detector, los objetos KF y SN
        :param MJD: double array, modified julian date
        :return: KF (filtro) y SN  (detector)
        """
        self.MJD = MJD
        if self.filter_type == 'kalman':
            KF = KalmanFilter(init_time=self.MJD[0] - 1.0)
        elif self.filter_type == 'MCC':
            KF = MaximumCorrentropyKalmanFilter(init_time=self.MJD[0] - 1.0)
        SN = SNDetector(flux_thres=self.flux_thres, vel_flux_thres=self.vel_flux_thres)
        return KF, SN

    def save_results(self, OB):
        """
        Guarda resultados despues de aplicar filtro en la segunda pasada.
        Si encuentra SNs
        :param OB: Observer object. Para realizar graficos
        :param results_dir: Directorio de resultados
        :return void:
        """
        filename = self.field + '-' + self.resultccd + '_NUO-' + str(self.NUO).zfill(2)
        if self.SN_index >= 0:
            filename = 'HiTS' + str(self.SN_index + 1).zfill(2) + '-' + ['nay', 'AYE'][self.SN_found] + '_' + filename
        if self.n_params > 0:
            filename = 'par-' + str(self.this_par).zfill(2) + '_' + filename
        np.savez(self.results_dir +  filename, objects=OB.obj)

    def decide_second_run(self, OB):
        """
        Decide si va a correr el algoritmo nuevamente para recuperar los objetos detectados.
        :param OB: Observer object
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


