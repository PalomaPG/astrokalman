import re
import os
import numpy as np
from astropy.io import fits


class DataPicker(object):

    def __init__(self, config_file):

        self.files_settings = {}
        self.data = {}
        self.mjd = []
        self.mjd_order = []

        with open(config_file, 'r') as f:
            #Ensure that line is written correctly
            for line in f:
                field, content = line.split(sep=' = ')
                self.files_settings[field] = content[:-1]

        self.set_data()

    def set_data(self):
        self.collect_data()
        self.select_science_images()
        self.select_images_dirs()

    def select_images_dirs(self):
        self.filter_obs()
        self.select_fits('diffDir')
        self.select_fits('invDir')
        self.select_npys('psfDir')
        self.select_npys('afluxDir', ref_dir ='scienceDir', init_index=0, n_pos=3, rest_len=0)

    def collect_data(self):
        self.data['baseDir'] = self.walking_through_files('baseRegEx', 'baseDir')
        self.data['maskDir'] = self.walking_through_files('maskRegEx', 'maskDir')
        self.data['crblastDir'] = self.walking_through_files('crblastRegEx', 'crblastDir')
        self.data['scienceDir'] = self.walking_through_files('scienceRegEx', 'scienceDir')

        self.data['diffDir'] = self.walking_through_files('diffRegEx', 'diffDir')
        self.data['invDir'] = self.walking_through_files('invRegEx', 'invDir')
        self.data['afluxDir'] = self.walking_through_files('afluxRegEx', 'afluxDir')
        self.data['psfDir'] = self.walking_through_files('psfRegEx', 'psfDir')

    def walking_through_files(self, regex, dir_):
        regex = re.compile(self.files_settings[regex])
        selected_base = []
        for root, dirs, files in os.walk(self.files_settings[dir_]):
            filtered_files = [root+'/'+f for f in filter(regex.search, files)]
            selected_base += filtered_files
        return selected_base

    def get_files(self, dir_key):
        return self.data[dir_key]

    def select_science_images(self):
        """
        listar mjd y airmass, hacer seleccion de acuerdo a estos valores
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

    def select_fits(self, dir_, print_=False):
        new_dir_content = []
        mjd_lst = []

        for image in self.data[dir_]:
            mjd = float(fits.open(image)[0].header['MJD-OBS'])
            if mjd in self.mjd:
                mjd_lst.append(mjd)
                new_dir_content.append(image)

        mjd_order = np.argsort(mjd_lst)
        self.data[dir_] = [new_dir_content[i] for i in mjd_order]

        if print_:
            print('# science images: %d;  # images in %s: %d' %
                (len(self.data['scienceDir']), dir_, len(self.data[dir_])))

    def select_npys(self, dir_, ref_dir='diffDir',
                      init_index=5, n_pos=5, rest_len=7):

        # for data including psf info: ref_dir = 'diffDir',init_index=5, n_pos=3, rest_len=7,
        # on the contrary ref_dir = 'scienceDir',init_index=0, n_pos=3, rest_len=0
        new_content = []

        for image in self.data[ref_dir]:
            image_prefix = (os.path.basename(os.path.normpath(image)))
            image_prefix = \
                image_prefix[init_index:[m.start() for m in re.finditer(r"_", image_prefix)][n_pos]+rest_len]
            for npy_file in self.data[dir_]:
                if npy_file.find(image_prefix) > -1:
                    new_content.append(npy_file)

    def obs_mjds(self):
        mjd_base = []
        mjd_msk = []
        mjd_crb = []

        for image in self.data['baseDir']:
            mjd_base.append(float(fits.open(image)[0].header['MJD-OBS']))

        for image in self.data['maskDir']:
            mjd_msk.append(float(fits.open(image)[0].header['MJD-OBS']))

        for image in self.data['crblastDir']:
            mjd_crb.append(float(fits.open(image)[0].header['MJD-OBS']))

        mjd = list(set(mjd_base) & set(mjd_msk) & set(mjd_crb))
        mjd.sort()
        return mjd

    def select_obs_images(self, mjd_lst, dir_):
        new_content = []
        for image in self.data[dir_]:
            mjd = (float(fits.open(image)[0].header['MJD-OBS']))
            if mjd in mjd_lst and np.around(mjd, 7) in self.mjd:
                new_content.append(image)

        self.data[dir_] = new_content

    def filter_obs(self):

        mjd_lst = self.obs_mjds()
        self.select_obs_images(mjd_lst, 'baseDir')
        self.select_obs_images(mjd_lst, 'maskDir')
        self.select_obs_images(mjd_lst, 'crblastDir')
        print('MJD list length: %d' % len(self.mjd))
        #self.mjd = list(set(self.mjd) & set(list(np.around(np.array(mjd_lst), 7))))
        #print('MJD list length: %d' %  len(self.mjd))


    def get_data(self):
        return self.data
