from .KalmanFilter import KalmanFilter
from .LinearPredict import LinearPredict
from .BasicCorrect import BasicCorrect

class BasicKalman(KalmanFilter):

    def __init__(self, image_size = (4094, 2046), sigma_a=0.1):
        KalmanFilter.__init__(self, image_size)
        self.ipredict = LinearPredict(sigma_a)
        self.icorrect = BasicCorrect()