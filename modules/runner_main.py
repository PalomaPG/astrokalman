from DataPicker import DataPicker
import pandas as pd
import sys


def main(sn_index_path, sn_index,  config_path):
    """
    :param sn_index_path: String. Path to input SN list
    :param sn_index: int. index of SN. It corresponds to the line of the SN depicted
                   in the file referenced by path
    :param path_: String. Path to input file which contains paths structures
    :return: void
    """
    SNs = pd.read_csv(sn_index_path, sep=',', header=0)
    sn_info = SNs.iloc[[sn_index]]
    picker = DataPicker(config_path, sn_info.iloc[0]['Semester'], sn_info.iloc[0]['Field'], sn_info.iloc[0]['CCD'])



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])