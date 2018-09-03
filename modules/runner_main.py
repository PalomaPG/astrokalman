from DataPicker import DataPicker
from FluxCalculator import *
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
    diff_ = picker.data['diffDir'][0]
    psf_ = picker.data['psfDir'][0]
    invvar_ = picker.data['invDir'][0]
    aflux_ = picker.data['afluxDir'][0]

    flux, flux_var, invvar = calc_fluxes(diff_, psf_, invvar_, aflux_)
    print(np.argwhere(flux>0))
    print(flux_var)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])