from abc import ABCMeta, abstractmethod

class IPredict(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, prev_time, curr_time):
        pass

