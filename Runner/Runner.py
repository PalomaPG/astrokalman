import re
import os
import numpy as np
from astropy.io import fits

from FluxCalculator import FluxCalculator


class Runner(object):

    def __init__(self, config_file):

        self.files_settings = {}
        self.data = {}

        with open(config_file, 'r') as f:
            #Ensure that line is written correctly
            for line in f:
                field, content = line.split(sep=' = ')
                self.files_settings[field] = content[:-1]

        self.set_data()

        #self.flux_calc = FluxCalculator()

    def set_data(self):

        self.data['baseDir'] = self.walking_through_files('baseRegEx', 'baseDir')
        self.data['maskDir'] = self.walking_through_files('maskRegEx', 'maskDir')
        self.data['crblastDir'] = self.walking_through_files('crblastRegEx', 'crblastDir')
        self.data['scienceDir'] = self.walking_through_files('scienceRegEx', 'scienceDir')

        self.data['psfDir'] = self.walking_through_files('psfRegEx', 'psfDir')
        self.data['diffDir'] = self.walking_through_files('diffRegEx', 'diffDir')
        self.data['invDir'] = self.walking_through_files('invRegEx', 'invDir')
        self.data['afluxDir'] = self.walking_through_files('afluxRegEx', 'afluxDir')

    def walking_through_files(self, regex, dir_):

        regex = re.compile(self.files_settings[regex])
        selected_base = []
        for root, dirs, files in os.walk(self.files_settings[dir_]):
            filtered_files = [root+'/'+f for f in filter(regex.search, files)]
            selected_base += filtered_files
        return selected_base

    def get_files(self, dir_key):
        return self.data[dir_key]

    def get_mjd_lst(self):
        return [float(fits.open(m)[0].header['MJD-OBS']) for m in self.data['scienceDir']]

    def get_airmass_lst(self):
        return [float(fits.open(m)[0].header['AIRMASS']) for m in self.data['scienceDir']]

    def select_data(self):
        pass

        """
        listar mjd y airmass, hacer seleccion de acuerdo a estos valores
        """