from abc import ABCMeta, abstractmethod


class IPredict(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, delta_t, state, state_cov, pred_state, pred_cov):
        pass

    def define_params(self, *args):
        pass

