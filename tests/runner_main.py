from modules.DataPicker import DataPicker
from KalmanFilter.BasicKalman import BasicKalman
from modules.utils import *
from modules.unix_time import *
import pandas as pd
import numpy as np
import sys


def main(obs_index_path, sn_index,  config_path):
    """
    :param sn_index_path: String. Path to input SN list
    :param sn_index: int. index of SN. It corresponds to the line of the SN depicted
                   in the file referenced by path
    :param path_: String. Path to input file which contains paths structures
    :return: void
    """
    obs = pd.read_csv(obs_index_path, sep=',', header=0)
    obs_info = obs.iloc[[sn_index]]
    #print(obs_info)
    picker = DataPicker(config_path, obs_info.iloc[0]['Semester'], obs_info.iloc[0]['Field'], obs_info.iloc[0]['CCD'])
    #print(picker.data['scienceDir'])
    diff_ = picker.data['diffDir']
    psf_ = picker.data['psfDir']
    invvar_ = picker.data['invDir']
    aflux_ = picker.data['afluxDir']


    t_flux = 0
    t_filter = 0

    for i in range(len(picker.mjd)):

        t = unix_time(calc_fluxes, (diff_[i], psf_[i], invvar_[i], aflux_[i],))
        t_flux = t['user'] + t_flux

        t = unix_time()

    print(t_flux)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])