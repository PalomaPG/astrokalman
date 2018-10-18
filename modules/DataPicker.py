import re
import os
import csv
import numpy as np
import pandas as pd
from astropy.io import fits


class DataPicker(object):
    """
    Collect data from specific directories according to CCD and field
    specified by the user in config_file

    """

    def __init__(self, config_file, semester, field_, ccd):
        """
        Object builder (aka. Constructor)
        :param config_file: Configuration file containing directories structures
                            in terms of regular expressions
        :param semester: String. Semester when the the observations were made (must be coherent to images name)
        :param field_: String.
        :param ccd: String. Camera ccd
        """

        self.files_settings = {}
        self.data = {}
        self.mjd = []
        self.mjd_order = []
        self.results_df = None
        self.field = field_
        self.ccd = ccd

        with open(config_file, 'r') as f:
            for line in f:
                field, content = line.split(sep=' = ')
                self.files_settings[field] = content[:-1]

        results_file = semester+'_'+str(field_).zfill(2)+'_'+ccd+'.csv'
        results_file = self.files_settings['results']+results_file
        if os.path.isfile(results_file):
            self.results_df = pd.read_csv(results_file, sep=',', header=0, index_col=0)

        self.config_reg_expressions(semester, field_, ccd)
        self.collect_data()
        self.filter_science_images()
        self.select_images()

    def config_reg_expressions(self, semester, field, ccd):

        """

        :param semester: String. Semester where observations where made. It must indicate
        the last two digits of the year and an 'A' or 'B' (character in capitals) to indicate if
        it refers to first or second semester respectively.
        :param field: Field number
        :param ccd:
        :return:
        """

        self.files_settings['maskDir'] = self.files_settings['maskDir'] % (semester, field, ccd)
        self.files_settings['scienceDir'] = self.files_settings['scienceDir'] % (semester, field, ccd)
        self.files_settings['diffDir'] = self.files_settings['diffDir'] % (semester, field, ccd)
        self.files_settings['psfDir'] = self.files_settings['psfDir'] % (semester, field, ccd)
        self.files_settings['invDir'] = self.files_settings['invDir'] % (semester, field, ccd)
        self.files_settings['afluxDir'] = self.files_settings['afluxDir'] % (semester, field, ccd)
        self.files_settings['maskRegEx'] = self.files_settings['maskRegEx'] % (semester, field, ccd)
        self.files_settings['scienceRegEx'] = self.files_settings['scienceRegEx'] % (semester, field, ccd)
        self.files_settings['diffRegEx'] = self.files_settings['diffRegEx'] % (semester, field, ccd)
        self.files_settings['invRegEx'] = self.files_settings['invRegEx'] % (semester, field, ccd)
        self.files_settings['afluxRegEx'] = self.files_settings['afluxRegEx'] % (semester, field, ccd)
        self.files_settings['psfRegEx'] = self.files_settings['psfRegEx'] % (semester, field, ccd)

    def collect_data(self):
        """

        :return:
        """

        self.data['scienceDir'] = self.walking_through_files('scienceRegEx', 'scienceDir')
        self.data['maskDir'] = self.walking_through_files('maskRegEx', 'maskDir')
        self.data['diffDir'] = self.walking_through_files('diffRegEx', 'diffDir')
        self.data['invDir'] = self.walking_through_files('invRegEx', 'invDir')
        self.data['afluxDir'] = self.walking_through_files('afluxRegEx', 'afluxDir')
        self.data['psfDir'] = self.walking_through_files('psfRegEx', 'psfDir')


    def walking_through_files(self, regex, dir_):
        """

        :param regex:
        :param dir_:
        :return:
        """
        regex = re.compile(self.files_settings[regex])
        selected_base = []
        for root, dirs, files in os.walk(self.files_settings[dir_]):
            filtered_files = [root+'/'+f for f in filter(regex.search, files)]
            selected_base += filtered_files
        return selected_base

    def filter_science_images(self):
        """
        Filters science images according to airmass: must be < 1.7
        :return: void.
        """

        data = []
        for fits_image in self.data['scienceDir']:
            with fits.open(fits_image) as opened_fits_image:
                if float(opened_fits_image[0].header['AIRMASS']) < 1.7:
                    data.append(fits_image)
                    self.mjd.append(float(opened_fits_image[0].header['MJD-OBS']))

        self.mjd_order = np.argsort(self.mjd)
        self.mjd.sort()
        self.data['scienceDir'] = [data[i] for i in self.mjd_order]

    def select_images(self):
        """

        :return:
        """
        self.select_fits('diffDir')
        self.select_fits('invDir')
        self.select_npys('psfDir')
        self.select_npys('afluxDir', ref_dir='scienceDir', init_index=0, n_pos=3, rest_len=0)

    def select_fits(self, dir_):
        """

        :param dir_:
        :return:
        """
        new_dir_content = []
        mjd_lst = []

        for image in self.data[dir_]:
            mjd = float(fits.open(image)[0].header['MJD-OBS'])
            if mjd in self.mjd:
                mjd_lst.append(mjd)
                new_dir_content.append(image)

        mjd_order = np.argsort(mjd_lst)
        self.data[dir_] = [new_dir_content[i] for i in mjd_order]

    def select_npys(self, dir_, ref_dir='diffDir', init_index=5, n_pos=5, rest_len=7):

        # for data including psf info: ref_dir = 'diffDir',init_index=5, n_pos=3, rest_len=7,
        # on the contrary ref_dir = 'scienceDir',init_index=0, n_pos=3, rest_len=0
        """

        :param dir_:
        :param ref_dir:
        :param init_index:
        :param n_pos:
        :param rest_len:
        :return:
        """
        new_content = []

        for image in self.data[ref_dir]:
            image_prefix = (os.path.basename(os.path.normpath(image)))
            image_prefix = \
                image_prefix[init_index:[m.start() for m in re.finditer(r"_", image_prefix)][n_pos]+rest_len]
            for npy_file in self.data[dir_]:
                if npy_file.find(image_prefix) > -1:
                    new_content.append(npy_file)

        self.data[dir_] = new_content

    def get_data(self):
        return self.data


    def verify_log(self):

        log_path = os.path.join(self.files_settings['scienceDir'], 'log.txt')
        if os.path.isfile(log_path):
            content = csv.DictReader(log_path) #np.genfromtxt(log_path, delimiter=':', dtype=object)
            pass
        else:
            pass
