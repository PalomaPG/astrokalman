import unittest

import numpy as np
from KalmanFilter.LinearKalman import LinearKalman

class TestKalmanLinearFilters(unittest.TestCase):

    def test_buildBasicFilter(self):
        image_size = (4094, 2046)
        flux = np.zeros(image_size)
        vflux = np.zeros(image_size)
        init_state = [flux, vflux]
        kalman = LinearKalman()
        self.assertTrue(kalman.predict(10.5, 12.0) == 1.5)

    def test_states(self):
        image_size = (4094, 2046)
        flux = np.ones(image_size)*7.3
        vflux = np.ones(image_size)*5
        init_state = [flux, vflux]
        kalman = LinearKalman()
        kalman.initialize_states(init_state=init_state)
        kalman.predict(5.6, 10.7)

if __name__ == '__main__':
    unittest.main()