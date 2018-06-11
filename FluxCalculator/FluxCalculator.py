import numpy as np
from astropy.io import fits
import scipy.ndimage as spn
import mahotas as mh

class FluxCalculator(object):

    def __init__(self):
        pass


"""
        print('Prepare base image')

        #self.base_image = fits.open(self.data_names['base'])
        self.base_image = fits.open(self.data_names['base'])[0].data
        self.base_mask = fits.open(self.data_names['mask_dq'])[0].data

        '''
        dil base mask es una imagen de 0s y 1s?. No se si es la version nueva de python 2.7, o del modulo dilate que no
        pesca el input
        '''
        #self.dil_base_mask = pm.dilate(self.base_mask > 0, B=np.ones((5, 5), dtype=bool))
        self.dil_base_mask = mh.dilate(self.base_mask > 0, Bc=np.ones((5, 5), dtype=bool))

        print('MJD...')
        MJD = [float(fits.open(m)[0].header['MJD-OBS']) for m in self.data_names['science']]
        # Order by MJD
        MJDOrder = np.argsort(MJD)
        MJD = np.array([MJD[i] for i in MJDOrder])

        print('Filter airmass')
        airmass = np.array([float(fits.open(m)[0].header['AIRMASS']) for m in self.data_names['science']])
        MJD = MJD[airmass < 1.7]
        MJDOrder = MJDOrder[airmass < 1.7]

        for e in ['science', 'diff', 'invVAR', 'psf', 'aflux']:
            if e in self.data_names:
            #if self.data_names.has_key(e):
                self.data_names[e] = [self.data_names[e][i] for i in MJDOrder]

        self.data_names['original_numFrames'] = len(MJD)
        self.data_names['original_MJD'] = MJD

        self.MJD = MJD
        print('End of get data names')
"""
