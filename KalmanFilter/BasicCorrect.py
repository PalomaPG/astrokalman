from .ICorrect import ICorrect

class BasicCorrect(ICorrect):

    def __init__(self):
        pass

    def correct(self, z, R,  pred_state,  pred_cov, state, state_cov):
        inv_S = pow(pred_cov[0, :] + R, -1)
        # Obtain Kalman Gain
        kalman_gain = pred_cov[[0, 1], :] * inv_S
        state = pred_state + kalman_gain * (z - pred_state[0, :])
        state_cov[[0, 1], :] = pred_cov[[0, 1], :] * (1.0 - kalman_gain[0, :])
        state_cov[2, :] = pred_cov[2, :] - kalman_gain[1, :] * pred_cov[1, :]

        return state, state_cov