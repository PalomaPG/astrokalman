from .IPredict import IPredict

class UnscentPredict(IPredict):

    def __init__(self):
        pass

    def predict(self, delta_t, state, state_cov, pred_state, pred_cov):
        return pred_state, pred_cov

    def sigma_points(self):
        pass