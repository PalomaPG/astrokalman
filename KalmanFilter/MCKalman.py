from .KalmanFilter import KalmanFilter
from .LinearPredict import LinearPredict
from .MCCorrect import MCCorrect

class MCKalman(KalmanFilter):

    def __init__(self, epsilon=1e-6, max_iter=10, silverman_sigma=False, image_size = (4094, 2046)):
        KalmanFilter.__init__(self, image_size)
        self.ipredict = LinearPredict()
        self.icorrect = MCCorrect(epsilon, max_iter, silverman_sigma)
