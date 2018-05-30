# -*- coding: utf-8 -*-

# SIF: Stream Images Filtering

import numpy as np
from glob import glob
from astropy.io import fits
import scipy.ndimage as spn
#import pymorph as pm
import mahotas as mh

class FITSHandler(object):

    def __init__(self, RD, accum_neg_flux_depth=4, accum_med_flux_depth=3):
        """
        Fits image manager. Se preocupa de recuperar todas las imagenes FITS.
        Y determina flujos
        :param RD: RunData instance. Toma los parametros que tiene la instancia de RunData
        :param accum_neg_flux_depth: Decision de umbral
        :param accum_med_flux_depth: Decicion de umbral
        """
        self.field = RD.field
        #print RD.ccd
        self.ccd = RD.ccd
        self.year = RD.year
        self.SN_index = RD.SN_index

        self.get_data_names()

        # Classifier criteria aspects
        self.accum_neg_flux_depth = accum_neg_flux_depth
        self.accum_neg_flux = np.zeros(tuple([self.accum_neg_flux_depth]) + RD.images_size, dtype=bool)

        self.accum_med_flux_depth = accum_med_flux_depth
        self.accum_median_flux = np.zeros(tuple([self.accum_med_flux_depth]) + RD.images_size)
        self.median_rejection = np.zeros(RD.images_size, dtype=bool)

    def get_data_names(self):
        """
        construccion de diccionario self.data_names. Direcciones de todas las imagenes en la maquina.
        Y construye arreglo de MJD de las observaciones.
        :return: Modified Julian day MJD
        """

        self.data_names = {}
        base_dir = '/home/apps/astro/data/DATA/'

        if glob('/home/pperez/Thesis/sif2/orig-sif'):  # At Leftraru
            print('At Leftraru')
            #print(glob(base_dir + 'Blind' + self.year + 'A_' + self.field+'/*/Blind*_image.fits.fz'))

            self.data_names['base'] = \
            sorted(glob(base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/Blind*_image.fits*'))[
                0]

            self.data_names['mask_dq'] = sorted(glob(base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/Blind*_dqmask.fits*'))[
                0]
            self.data_names['base_crblaster'] = sorted(glob(
                base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/Blind*image_crblaster.fits*'))[0]
            # projection
            self.data_names['science'] = sorted(glob(
                base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/Blind*image_crblaster_grid02_lanczos2.fits'))

            self.data_names['diff'] = []
            self.data_names['invVAR'] = []
            self.data_names['psf'] = []
            self.data_names['aflux'] = []

            for science_filename in self.data_names['science']:
                ind = science_filename.find('_image_')
                epoch = science_filename[ind - 2:ind]
                # difference image
                self.data_names['diff'] += [np.sort(glob(
                    base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/Diff*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[
                                                0]]
                # 1/var pf diff image
                self.data_names['invVAR'] += [np.sort(glob(
                    base_dir + 'Blind' + self.year + 'A_' + self.field + '/*/invVAR*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[
                                                  0]]
                print('PSF...')
                print(glob(
                    '/home/apps/astro/data/SHARED/Blind' + self.year + 'A_' +
                    self.field + '/' + self.ccd + '/CALIBRATIONS/psf*' + self.ccd + '_' + epoch + '*grid02*'))

                # diff image psf
                self.data_names['psf'] += [np.sort(glob(
                    '/home/apps/astro/data/SHARED/Blind' + self.year + 'A_' + self.field + '/' + self.ccd + '/CALIBRATIONS/psf*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[
                                               0]]

                # astrometric and relative flux constants
                self.data_names['aflux'] += [np.sort(glob(
                    '/home/apps/astro/data/SHARED/Blind' + self.year + 'A_' + self.field + '/' + self.ccd + '/CALIBRATIONS/match_*' + epoch + '-02.npy')).tolist()[
                                                 0]]
        else:

            print('At Pablo')

            # baseDir = '/run/media/tesla/Almacen/Huentelemu/R20' + year + 'CCDs/HiTS' + str(snIndex).zfill(2) + 'SN/'
            baseDir = '/home/paloma/Documents/Memoria/data/Blind15A_38/S25/' # + str(self.SN_index + 1).zfill(2) + 'SN/'
            # baseDir = 'C:/Users/Bahamut/Desktop/HiTS' + str(self.SN_index+1).zfill(2) + 'SN/'
            # baseDir = 'D:/Lab Int Comp/R2015CCDs/HiTS' + str(self.SN_index+1).zfill(2) + 'SN/'

            self.data_names['base'] = glob(baseDir+'Blind15A_38_S25_*_image.fits.fz')[0]
            self.data_names['mask_dq'] = glob(baseDir+'Blind15A_38_S25_*_dqmask.fits.fz')[0]

            self.data_names['base_crblaster'] = glob(baseDir+'Blind15A_38_S25_*_image_crblaster.fits')[0]
            self.data_names['science'] = glob(baseDir+'Blind15A_38_S25_*_image_crblaster_grid02_lanczos2.fits')

            self.data_names['diff'] = []
            self.data_names['psf'] = []
            self.data_names['invVAR'] = []
            self.data_names['aflux'] = []

            for science_filename in self.data_names['science']:
                ind = science_filename.find('_image_')
                epoch = science_filename[ind - 2:ind]
                #print epoch

                self.data_names['diff'] += [np.sort(glob(baseDir + 'Diff*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[0]]

                # [baseDir+'Diff_Blind15A_38_S25_03-02t_grid02_lanczos2.fits']
                self.data_names['psf'] += [np.sort(glob(baseDir + 'CALIBRATION/psf*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[0]]
                #print self.data_names['psf']
                #[baseDir+'CALIBRATION/'+'psf_Blind15A_38_S25_03-02t_grid02_lanczos2.npy']
                self.data_names['invVAR'] += [np.sort(glob(baseDir + 'invVAR*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[0]]
                #[baseDir+'invVAR_Blind15A_38_S25_03-02t_grid02_lanczos2.fits']
                self.data_names['aflux'] += [np.sort(glob(baseDir + 'CALIBRATION/match_*' + epoch + '-02.npy')).tolist()[0]]

                #[baseDir+'CALIBRATION/'+'match_Blind15A_38_S25_03-02.npy']

            print('At Paloma')

            # baseDir = '/run/media/tesla/Almacen/Huentelemu/R20' + year + 'CCDs/HiTS' + str(snIndex).zfill(2) + 'SN/'
            baseDir = '/home/paloma/Documents/Memoria/data/Blind15A_38/S25/' # + str(self.SN_index + 1).zfill(2) + 'SN/'
            # baseDir = 'C:/Users/Bahamut/Desktop/HiTS' + str(self.SN_index+1).zfill(2) + 'SN/'
            # baseDir = 'D:/Lab Int Comp/R2015CCDs/HiTS' + str(self.SN_index+1).zfill(2) + 'SN/'

            self.data_names['base'] = glob(baseDir+'Blind15A_38_S25_*_image.fits.fz')[0]
            self.data_names['mask_dq'] = glob(baseDir+'Blind15A_38_S25_*_dqmask.fits.fz')[0]

            self.data_names['base_crblaster'] = glob(baseDir+'Blind15A_38_S25_*_image_crblaster.fits')[0]
            self.data_names['science'] = glob(baseDir+'Blind15A_38_S25_*_image_crblaster_grid02_lanczos2.fits')

            self.data_names['diff'] = []
            self.data_names['psf'] = []
            self.data_names['invVAR'] = []
            self.data_names['aflux'] = []

            for science_filename in self.data_names['science']:
                ind = science_filename.find('_image_')
                epoch = science_filename[ind - 2:ind]
                #print epoch

                self.data_names['diff'] += [np.sort(glob(baseDir + 'Diff*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[0]]

                # [baseDir+'Diff_Blind15A_38_S25_03-02t_grid02_lanczos2.fits']
                self.data_names['psf'] += [np.sort(glob(baseDir + 'CALIBRATION/psf*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[0]]
                #print self.data_names['psf']
                #[baseDir+'CALIBRATION/'+'psf_Blind15A_38_S25_03-02t_grid02_lanczos2.npy']
                self.data_names['invVAR'] += [np.sort(glob(baseDir + 'invVAR*' + self.ccd + '_' + epoch + '*grid02*')).tolist()[0]]
                #[baseDir+'invVAR_Blind15A_38_S25_03-02t_grid02_lanczos2.fits']
                self.data_names['aflux'] += [np.sort(glob(baseDir + 'CALIBRATION/match_*' + epoch + '-02.npy')).tolist()[0]]

                #[baseDir+'CALIBRATION/'+'match_Blind15A_38_S25_03-02.npy']

        # Prepare base image

        #self.base_image = fits.open(self.data_names['base'])
        self.base_image = fits.open(self.data_names['base'])[0].data
        self.base_mask = fits.open(self.data_names['mask_dq'])[0].data

        '''
        dil base mask es una imagen de 0s y 1s?. No se si es la version nueva de python 2.7, o del modulo dilate que no
        pesca el input
        '''
        #self.dil_base_mask = pm.dilate(self.base_mask > 0, B=np.ones((5, 5), dtype=bool))
        self.dil_base_mask = mh.dilate(self.base_mask > 0, Bc=np.ones((5, 5), dtype=bool))

        MJD = [float(fits.open(m)[0].header['MJD-OBS']) for m in self.data_names['science']]
        # Order by MJD
        MJDOrder = np.argsort(MJD)
        MJD = np.array([MJD[i] for i in MJDOrder])

        # Filter airmass
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

    def naylor_photometry(self, invvar):
        """
        Convoluciona flujos. Calcula flujo y su varianza
        :param invvar: float matrix, Varianza de la imagen de diferencia
        :return void:
        """
        input_1 = self.diff * invvar
        #print 'dimension input 1 @ naylor_photometry %d' % input_1.ndim
        #print 'dimension psf @ naylor_photometry %d' % self.psf.ndim
        if input_1.ndim != self.psf.ndim:
            return

        self.flux = spn.convolve(input_1, self.psf)
        psf2 = self.psf ** 2
        convo = spn.convolve(invvar, psf2)
        convo[convo == 0] = 0.000001
        self.var_flux = 1 / convo
        self.flux = self.flux * self.var_flux

    def load_fluxes(self, o):
        """
        Calcula datos de los flujos
        :param o: int, indice de observacion
        :return void:
        """

        self.science = fits.open(self.data_names['science'][o])[0].data
        self.diff = fits.open(self.data_names['diff'][o])[0].data
        self.psf = np.load(self.data_names['psf'][o])
        invvar = fits.open(self.data_names['invVAR'][o])[0].data

        # Filter bad invVAR values
        invvar[invvar == np.inf] = 0.01

        self.naylor_photometry(invvar)

        # Aflux Correction
        if (self.data_names['diff'][o].find('02t') > 0):
            aflux = np.load(self.data_names['aflux'][o])
            aflux = aflux[0]
            self.flux = self.flux / aflux
            self.var_flux = self.var_flux / (aflux * aflux)

        self.var_flux = np.sqrt(self.var_flux)

        # Filter nan fluxes
        self.flux[np.isnan(self.flux)] = 0.001

        # Register negative fluxes (bool cube), Matriz dim_x x dim_y x # observations
        # Flujo negativo. Pixeles que tuvieron valores negativos
        # Queremos descartar todos los candidatos que estuvieron cerca de un pixel
        # negativo, ya que suelen ser candidatos de malas restas.
        self.accum_neg_flux[o % self.accum_neg_flux_depth, :] = self.flux < 0

        # Hasta cierta cantidad de observaciones (accum_med_flux_depth), guarda valores de flujo
        # para calcular la mediana de las imagenes de flujo.
        # Los pixeles que tienen una mediana de flujo mayor a 1500, son descartados.
        # Accumulate fluxes for high initial median rejection
        if o < self.accum_med_flux_depth:
            self.accum_median_flux[o, :] = self.flux
        elif o == self.accum_med_flux_depth: # Cubo de booleans
            self.median_rejection = np.median(self.accum_median_flux, 0) > 1500.0

        # Hay pixeles que tienen valores muy altos de flujo en toda la corrida (el algoritmo puede 'confundirse'
        # considerando la existencia de SN).

