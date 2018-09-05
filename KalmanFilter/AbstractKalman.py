import numpy as np

from abc import ABC, abstractclassmethod

class AbstractKalman(ABC):

    def __init__(self, image_size):
        self.image_size = image_size
        self.state = np.zeros((2, ) + self.image_size, dtype=float)
        self.state_cov = np.zeros((3, ) + image_size, dtype=float)

    def initialize_states(self, init_state):
        self.state[0] = init_state[0]
        self.state[1] = init_state[1]

    @abstractclassmethod
    def predict(self):
        pass

    @abstractclassmethod
    def correct(self,  pred_state, pred_cov):
        pass

    def update(self):
        self.predict()
        self.correct()