from .ICorrect import ICorrect
from modules.unscented_utils import *


class UnscentCorrect(ICorrect):

    def __init__(self, f_func, h_func, Wm, Wc, lambda_,  d=2):

        self.d = d
        self.lambda_ = lambda_
        self.h_func = h_func
        self.f_func = f_func
        self.Wm = Wm
        self.Wc = Wc

    def define_params(self, *args):
        if len(args) == 3:
            self.Xs = args[0]
            self.delta_t = args[1]
            self.image_size = args[2]
        elif len(args) == 2:
            self.Xs = args[2]
            self.delta_t = args[1]
        elif len(args) == 1:
            self.delta_t =args[1]

    def correct(self, z, R, pred_state, pred_cov, state, state_cov):
        Ys = sigma_points(pred_state, pred_cov, lambda_=self.lambda_, N=self.d)
        #print('-----------Ys-----------------')
        #print(Ys)
        #print('-------------------------------')
        state, state_cov = propagate_func(self.h_func, self.Wm, self.Wc, Ys, self.delta_t) # z^, S_k - R

        #Residual
        residual = np.zeros(shape=state.shape)
        residual[0] = z - state[0]

        #Innovation
        S=np.zeros(shape=state_cov.shape)
        S[0] = state_cov[0] + R
        S[1] = state_cov[1]
        S[2] = state_cov[2]

        #State-measurement
        h_diff = []
        f_diff = []

        D = 2*self.d+1

        for i in range(D):
            f_diff.append(perform(self.f_func, self.Xs[i], (self.delta_t,)))
            h_diff.append(perform(self.h_func, Ys[i], ()))

        for i in range(D):
            h_diff[i] = h_diff[i] - state
            f_diff[i] = f_diff[i] - pred_state

        C = multiple_dot_products(h_diff, h_diff, self.Wc, image_size=self.image_size)
        ##Optimal gain
        K = matrices_dot_product(C, matrix_inverse(S))
        state = pred_state + matrix_vector_dot_product(K, residual)
        #print(state)
        state_cov = np.zeros(shape=state_cov.shape)
        state_cov[0] = 100
        state_cov[2] = 20
        #state_cov = (pred_cov - matrices_dot_product(K, matrices_dot_product(S, K)))
        #pred_cov - matrices_dot_product(K, matrices_dot_product(S, K))#K @ S @ K.T

        return state, state_cov

