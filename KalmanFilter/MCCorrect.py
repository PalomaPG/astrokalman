from modules.utils import cholesky

import numpy as np
from .ICorrect import ICorrect

class MCCorrect(ICorrect):

    def __init__(self, epsilon=1e-6, max_iter=10, silverman_sigma=False):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.silverman_sigma = silverman_sigma

    def correct(self, pred_state, pred_cov, flux, flux_cov, state, state_cov):
        chol_p, inv_chol_p = cholesky(pred_cov)
        prev_iter_state = pred_state.copy()

        i = 1

        while True:
            C = np.concatenate((pred_state, np.expand_dims(flux, 0))) - prev_iter_state[[0, 1, 0], :]