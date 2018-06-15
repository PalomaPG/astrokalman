import re
import os
import numpy as np
from astropy.io import fits


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

    def set_data(self):
        self.collect_data()
        self.select_science_images()
        self.select_images_dirs()

    def select_images_dirs(self):
        self.select_images('baseDir')
        self.select_images('maskDir')
        self.select_images('crblastDir')

        self.select_images('diffDir')
        self.select_images('invDir')
        self.select_images('afluxDir')
        self.select_images('psfDir')

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
        data_science = []

        for fits_image in self.data['scienceDir']:
            with fits.open(fits_image) as opened_fits_image:
                if float(opened_fits_image[0].header['AIRMASS']) < 1.7:
                    data_science.append(fits_image)

        self.data['scienceDir'] = data_science
    # Append in order

    def select_images(self, dir_):
        new_dir = []

        for science_image in self.data['scienceDir']:

            science_prefix = (os.path.basename(os.path.normpath(science_image)))
            science_prefix = science_prefix[:[m.start() for m in re.finditer(r"_", science_prefix)][2]+3]
            print(fits.open(science_image)[0].header['MJD-OBS'], science_prefix)
            isin = False
            for fits_file in self.data[dir_]:
                if fits_file.find(science_prefix) > -1:
                    new_dir.append(fits_file)
                    isin = True
                    break
            if not isin:
                #print(science_image)
                self.data['scienceDir'].remove(science_image)

        print("Science image number: %d, number of resultant images in directory %s: %d"
             % (len(self.data['scienceDir']), dir_, len(new_dir)))

        self.data[dir_] = new_dir