# -*- coding: utf-8 -*-

# SIF: Stream Images Filtering

import numpy as np
#import pymorph as pm
import mahotas as mh

class SNDetector(object):

    def __init__(self, n_consecutive_alerts=4, images_size=(4094, 2046), flux_thres=500.0, vel_flux_thres=150.0,
                 vel_satu=3000.0):
        """
        An object that defines a supernova detector. Guarda todos los umbrales y trabaja con el KF y FH.
        Prepara coordenadas de candidatos (guarda en lista CandData).

        :param n_consecutive_alerts: # alertas consecutivas, para que un objeto sea considerado una supernova con seguridad.
        :param images_size: int tuple,  dimensiones en pixeles (x, y)
        :param flux_thres: float, umbral de flujo
        :param vel_flux_thres: float, umbral  de velocidad de flujo
        :param vel_satu: float, saturacion ...
        """
        self.n_conditions = 7 # # de condiciones para considerar un pixel como parte de una SN
        self.n_consecutive_alerts = n_consecutive_alerts
        self.pixel_conditions = np.zeros(tuple([self.n_conditions]) + images_size, dtype=bool)
        # guarda la info para cada pixel (cuales condiciones cumple cada pixel)
        self.pixel_flags = np.zeros(images_size, dtype=int)
        #  Suma binaria de pixel conditions
        self.accum_compliant_pixels = np.zeros(tuple([self.n_consecutive_alerts]) + images_size, dtype=bool)
        #
        self.CandData = []

        self.flux_thres = flux_thres
        self.vel_flux_thres = vel_flux_thres
        self.vel_satu = vel_satu

    def subsampled_median(self, image, sampling):
        """
        Obtiene la mediana de subimagenes de image
        :param image: float matrix, Imagen de entrada
        :param sampling: int,
        :return:
        """
        size1 = 4094
        size2 = 2046
        margin = 100
        yAxis = range(margin, size1 - margin, sampling)
        xAxis = range(margin, size2 - margin, sampling)
        sampled_image = np.zeros((len(yAxis), len(xAxis)))
        x = 0
        for i in yAxis:
            y = 0
            for j in xAxis:
                sampled_image[x, y] = image[i, j]
                y += 1
            x += 1
        return np.median(sampled_image)

    def pixel_discrimination(self, o, FH, KF):
        """
        Discrimiancion de pixeles por separado
        :param o:  Indice de observacion
        :param FH: FitsHandler instance
        :param KF: KalmanFilter instance
        :return: void
        """
        # obtencion de mediana de imagen science (tomando submuestras)
        epoch_science_median = self.subsampled_median(FH.science, 20) # float

        self.pixel_conditions[:] = False
        self.pixel_flags[:] = 0

        # Si el flujo estimado por KF es mayor al umbral de flujo
        self.pixel_conditions[0, :] = KF.state[0, :] > self.flux_thres
        #print '----------------------------------------------------'
        #print self.flux_thres
        #print np.isnan(KF.state[1, :]).any()
        #print '----------------------------------------------------'

        # Velocidad de flujo estimada mayor a umbral de velocidad de flujo
        self.pixel_conditions[1, :] = KF.state[1, :] > self.vel_flux_thres * (
                    self.vel_satu - np.minimum(KF.state[0, :], self.vel_satu)) / self.vel_satu

        self.pixel_conditions[2, :] = FH.science > epoch_science_median + 5 # umbral estimado para considerar
        #  pixeles + brillantes que el cielo

        # DESCARTE DE PIXELES DEFECTUOSOS
        self.pixel_conditions[3, :] = KF.state_cov[0, :] < 150.0 # varianza de  flujo
        self.pixel_conditions[4, :] = KF.state_cov[2, :] < 150.0 # varianza de vel. flujo
        self.pixel_conditions[5, :] = np.logical_not(FH.dil_base_mask) #
        self.pixel_conditions[6, :] = np.logical_not(FH.median_rejection) #

        for i in range(self.n_conditions):
            self.pixel_flags[np.logical_not(self.pixel_conditions[i, :])] += 2 ** i

        self.accum_compliant_pixels[o % self.n_consecutive_alerts, :] = self.pixel_flags == 0

    def neighboring_pixels(self):
        """

        :return:
        """

        self.PGData = {}  # Pixel group data
        self.PGData['pixel_coords'] = []

        alert_pixels = np.all(self.accum_compliant_pixels, 0)

        if not np.any(alert_pixels):
            self.PGData['mid_coords'] = np.zeros((0, 2), dtype=int)
            return

        #labeled_image = pm.label(alert_pixels, Bc=np.ones((3, 3), dtype=bool))
        labeled_image, nr_objects = mh.label(alert_pixels, Bc=np.ones((3, 3), dtype=int))
        print(labeled_image[0])

        LICoords = np.nonzero(labeled_image)
        LIValues = labeled_image[LICoords]
        LICoords = np.array(LICoords).T

        sortedArgs = np.argsort(LIValues)
        LIValues = LIValues[sortedArgs]
        LICoords = LICoords[sortedArgs, :]

        n_neighboring_pixels = LIValues[-1]

        self.PGData['mid_coords'] = np.zeros((n_neighboring_pixels, 2), dtype=int)

        for i in range(n_neighboring_pixels):
            self.PGData['pixel_coords'] += [LICoords[LIValues == i + 1, :]]
            self.PGData['mid_coords'][i, :] = np.round(np.mean(self.PGData['pixel_coords'][i], 0))

    def filter_groups(self, FH, KF):
        """

        :param FH:
        :param KF:
        :return:
        """
        n_pixel_groups = self.PGData['mid_coords'].shape[0]

        self.PGData['group_flags'] = np.zeros(n_pixel_groups, dtype=int)
        self.PGData['group_flags_map'] = -np.ones((4094, 2046), dtype=int)

        for i in range(n_pixel_groups):
            posY, posX = self.PGData['mid_coords'][i, :]

            # Discard groups with negative flux around (bad substractions)
            NNFR = 4
            a, b = posY - NNFR, posY + NNFR + 1
            c, d = posX - NNFR, posX + NNFR + 1
            if np.any(FH.accum_neg_flux[:, a:b, c:d]):
                self.PGData['group_flags'][i] += 1

            # Local Maxima Radius in Science Image
            LMSR = 3
            a, b = posY - LMSR + 1, posY + LMSR + 2
            c, d = posX - LMSR + 1, posX + LMSR + 2
            #scienceLM = pm.regmax(FH.science[a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
            scienceLM = mh.regmax(FH.science[a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
            if not np.any(scienceLM[1:-1, 1:-1]):
                self.PGData['group_flags'][i] += 2

            # Local Maxima Radius in Flux Image
            LMSR = 3
            a, b = posY - LMSR + 1, posY + LMSR + 2
            c, d = posX - LMSR + 1, posX + LMSR + 2
            #fluxLM = pm.regmax(FH.flux[a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
            fluxLM = mh.regmax(FH.flux[a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
            if not np.any(fluxLM[1:-1, 1:-1]):
                self.PGData['group_flags'][i] += 4

            # Local Maxima Radius in Estimated Flux Velocity Image
            LMSR = 3
            a, b = posY - LMSR + 1, posY + LMSR + 2
            c, d = posX - LMSR + 1, posX + LMSR + 2
            #velLM = pm.regmax(KF.state[1, a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
            velLM = mh.regmax(KF.state[1, a:b, c:d].astype(int), Bc=np.ones((3, 3), dtype=bool))
            if not np.any(velLM[1:-1, 1:-1]):
                self.PGData['group_flags'][i] += 8

            # Above local science median
            ASMR = 3
            a, b = posY - ASMR, posY + ASMR + 1
            c, d = posX - ASMR, posX + ASMR + 1
            if not (FH.science[posY, posX] > np.median(FH.science[a:b, c:d]) + 15):
                self.PGData['group_flags'][i] += 16

            # Brightest Pixel on stamps (flux and science)
            BPOS = 10
            a, b = posY - BPOS, posY + BPOS + 1
            c, d = posX - BPOS, posX + BPOS + 1
            brightPixels = np.logical_or(FH.flux[a:b, c:d] > 2 * FH.flux[posY, posX],
                                         FH.science[a:b, c:d] > 2 * FH.science[posY, posX])
            if np.any(brightPixels):
                self.PGData['group_flags'][i] += 32

            # Center over mask
            if FH.base_mask[posY, posX] > 0:
                self.PGData['group_flags'][i] += 64

            # Center over median-rejected pixel
            if FH.median_rejection[posY, posX]:
                self.PGData['group_flags'][i] += 128

            # flux variance
            if FH.var_flux[posY, posX] > 250.0:
                self.PGData['group_flags'][i] += 256

            self.PGData['group_flags_map'][self.PGData['pixel_coords'][i][:, 0], self.PGData['pixel_coords'][i][:, 1]] = \
            self.PGData['group_flags'][i]

    def draw_complying_pixel_groups(self, o, FH, KF):
        """
        Forma grupos de pixeles que cumple las condiciones individualmente
        1. Hace pasar por separado por umbrales
        2. ..por grupos
        :param o: Indice de observacion
        :param FH: instancia de FitsHandler
        :param KF: instancia de KalmanFilter
        :return: void
        """

        # Discriminate every pixel by itself
        self.pixel_discrimination(o, FH, KF)

        # Determine groups of neighboring compliant pixels
        self.neighboring_pixels()

        # Filter groups by morphological analysis
        self.filter_groups(FH, KF)

        print(' Pixel Groups: ' + str(self.PGData['mid_coords'].shape[0]))
        print(' Filtered Pixel Groups: ' + str(len(np.nonzero(self.PGData['group_flags'] == 0)[0])))

    def update_candidates(self, o):
        """

        :param o:
        :return:
        """
        cand_mid_coords = self.PGData['mid_coords'][self.PGData['group_flags'] == 0, :]

        for i in range(cand_mid_coords.shape[0]):

            for c in range(len(self.CandData) + 1):
                if c == len(self.CandData):
                    # New candidate
                    new_cand = {}
                    new_cand['coords'] = cand_mid_coords[i, :]
                    new_cand['epochs'] = [o]
                    self.CandData += [new_cand]
                else:
                    # Part of a previous candidate?
                    if (np.sqrt(np.sum((cand_mid_coords[i, :] - self.CandData[c]['coords']) ** 2)) < 4.0):
                        n_epochs = len(self.CandData[c]['epochs'])
                        self.CandData[c]['coords'] = (self.CandData[c]['coords'] * n_epochs + cand_mid_coords[i, :]) / (
                                    n_epochs + 1)
                        self.CandData[c]['epochs'] += [o]
                        break

    def check_candidates(self, RD):
        """

        :param RD: RunData instance
        :return:
        """

        RD.NUO = 0  # Number of Unknown Objects

        if RD.SN_index >= 0:
            RD.SN_found = False

            for i in range(len(self.CandData)):
                distance = np.sqrt(np.sum((self.CandData[i]['coords'] - RD.SN_pos) ** 2))

                if distance < 4.0:
                    self.CandData[i]['status'] = 0
                    RD.SN_found = True
                else:
                    self.CandData[i]['status'] = -1
                    RD.NUO += 1
        else:
            RD.NUO = len(self.CandData)
            for i in range(len(self.CandData)):
                self.CandData[i]['status'] = -1

        RD.CandData = self.CandData
