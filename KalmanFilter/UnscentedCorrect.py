from .ICorrect import ICorrect
from modules.unscented_utils import *


class UnscentedCorrect(ICorrect):

    def __init__(self, f_func, h_func, f_args, h_args,  Wm, Wc, lambda_,  d=2, image_size= (4094, 2046)):

        self.d = d
        self.lambda_ = lambda_
        self.h_func = h_func
        self.f_func = f_func
        self.Wm = Wm
        self.Wc = Wc
        self.image_size = image_size
        self.f_args =f_args
        self.h_args =h_args

    def define_params(self, Xs):
        self.Xs = Xs

    def correct(self, z, R, pred_state, pred_cov, state, state_cov, delta_t):
        Ys = sigma_points(pred_state, pred_cov, lambda_=self.lambda_, N=self.d)
        pred_z = propagate_func_corr(self.h_func, self.Wm, self.Wc, Ys, delta_t,
                                          args=self.h_args,image_size=self.image_size) # z^, S_k - R

        residuals = np.zeros(shape=state.shape)
        residuals[0] = z - pred_z[0]
        residuals[1] = pred_z[1]

        #Innovation
        S_innovation = propagate_func_corr(self.h_func, self.Wm, self.Wc, Ys, delta_t,
                                          args=self.h_args,image_size=self.image_size,
                                          mean=False)
        S_innovation[0] = S_innovation[0] + R

        # Cross covariance matrix
        C = cross_covariance(self.f_func, self.h_func, self.Wc, self.Xs, Ys, delta_t, self.f_args,
                         self.h_args, pred_state, pred_z, image_size=self.image_size)



        K = optimal_gain(C, S_innovation, image_size=self.image_size)


        k = np.zeros((tuple([2])+self.image_size))
        k[0] = K[0]*residuals[0] + K[1]*residuals[1]
        k[1] = K[2]*residuals[0] + K[3]*residuals[1]
        state = pred_state + k
        print('Calc state...\n')
        print(pred_state)
        print(k)

        KSKt = get_KSKt_product(K, S_innovation, image_size=self.image_size)
        state_cov = pred_cov - KSKt

        return state, state_cov, K[[0, 3]]


