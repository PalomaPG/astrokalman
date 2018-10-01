from abc import ABCMeta, abstractmethod

class ICorrect(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def correct(self, pred_state,  pred_cov):
        pass