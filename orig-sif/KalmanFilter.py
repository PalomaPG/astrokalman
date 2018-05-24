# -*- coding: utf-8 -*-

# SIF: Stream Images Filtering

import numpy as np


class KalmanFilter(object):

    def __init__(self, init_time=0.0, init_state=0.0, images_size=(4094, 2046), initial_variance=100.0, sigma=1000.0,
                 std_factor=100.0):
        """
        Instancia que encapsula parametros usados en filtro de Kalman
        :param init_time: float, primer valor de MJD, por defecto 0.0) tiempo inicial de las observaciones
        :param init_state: float, priver valor de flujo de todas las imagenes, correspondiente al MJD
        :param images_size: int array, tama~nos de la imagen (numero pixeles por dimension)
        :param initial_variance: float, varianza inicial que van a tener las matrices de covarianza.
        :param sigma: float, valor que se ocupa para el filtro de correntropia
        :param std_factor: float, factor
        """
        self.num_states = 2
        # numero de valores en el triangulo superior en una matriz
        # (queremos guardar 3 de la matriz de covarianza).
        self.num_cov_elements = self.num_states * (self.num_states + 1) / 2

        # Tiempo incial de observacion
        self.time = init_time

        # float cube, Guarda flujo y velocidad de flujo de cada pixel
        # (estado estimado)
        self.state = init_state * np.ones(tuple([self.num_states]) + images_size)

        # Preparacion de covarianza estimada (3 varianzas): flujo, entre flujo y velocidad de flujo,
        #  y varianza de velocidad
        self.state_cov = np.zeros(tuple([self.num_cov_elements]) + images_size)
        # Se asume inicialmente independencia, pero el filtro determinara si hay o no correlacion.
        self.state_cov[[0, 2], :] = initial_variance

        # Estado predicho para la primer observacion (asume velocidad de flujo como 0, en este caso
        # seria igual la prediccion es igual a la estimacion)
        self.pred_state = self.state.copy()
        self.pred_state_cov = self.state_cov.copy()


        self.sigma = sigma
        self.std_factor = std_factor
        self.observation = 0 # se asume primer indice de observacion como 0

    def variable_accel_Q(self, delta_t, sigma_a=0.1):
        """
        Cuanto crece incerteza segun cuanto tiempo se deja pasar desde la ultima observacion.
        :param delta_t: float, Paso del tiempo
        :param sigma_a: float, valor arbitrario
        :return: float array
        """
        return np.array([delta_t ** 4 / 4, delta_t ** 3 / 2, delta_t ** 2]) * (sigma_a ** 2)

    def predict_at_new_time(self, new_time):
        """
        Prepara imagen de prediccion a partir de estados estimados en un nuevo tiempo new_time
        :param new_time: float, MJD
        :return: void
        """
        # Obtain delta_t (tiempo desde la ultima observacion)
        delta_t = new_time - self.time

        # Predict mean (calculo del flujo y la varianza del flujo predichas)
        # 0 : flujo
        # 1 : velocidad de flujo
        self.pred_state[0, :] = self.state[0, :] + self.state[1, :] * delta_t
        self.pred_state[1, :] = self.state[1, :].copy() # supuesto: velocidad constante en esta proyeccion

        # Predict covariance,
        Q = self.variable_accel_Q(delta_t)
        # Se calcula reserve que se usa 2 veces.
        reserve = self.state_cov[1, :] + delta_t * self.state_cov[2, :]
        self.pred_state_cov[0, :] = self.state_cov[0, :] + delta_t * (self.state_cov[1, :] + reserve) + Q[0]
        self.pred_state_cov[1, :] = reserve + Q[1]
        self.pred_state_cov[2, :] = self.pred_state_cov[2, :] + Q[2]

    def correct_with_measurements(self, z, R):
        """
        Corrije con las mediciones (observaciones), para obtener nuevos estados estimados
        :param z: float matrix, flujo
        :param R: float matrix, varianza de flujo
        :return:
        """
        # Obtain inverse of residual's covariance
        inv_S = pow(self.pred_state_cov[0, :] + R, -1)

        # Obtain Kalman Gain
        self.kalman_gain = self.pred_state_cov[[0, 1], :] * inv_S

        # Correct estimate mean
        self.state = self.pred_state + self.kalman_gain * (z - self.pred_state[0, :])

        # Correct estimated covariance (Optimal gain version)
        self.state_cov[[0, 1], :] = self.pred_state_cov[[0, 1], :] * (1.0 - self.kalman_gain[0, :])
        self.state_cov[2, :] = self.pred_state_cov[2, :] - self.kalman_gain[1, :] * self.pred_state_cov[1, :]

    def update(self, new_time, FH):
        """
        Iteracion que predice estados de Kalman en un tiempo new_time. Compara la prediccion obtenida
        con las nuevas observaciones contenidas en FH, para finalmente actualizar los estados estimados.
        :param new_time: float, MJD
        :param FH: FitsHandler instance.
        :return: void
        """
        # Prediction
        self.predict_at_new_time(new_time)

        # Correction: Solo flujo y varianza de flujo
        self.correct_with_measurements(FH.flux, FH.var_flux)

        # Update time of estimations
        self.time = new_time

        # Update observation index
        self.observation += 1