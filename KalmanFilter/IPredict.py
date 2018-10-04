from abc import ABCMeta, abstractmethod


class IPredict(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, prev_time, curr_time, state, state_cov, pred_state, pred_cov):
        pass

