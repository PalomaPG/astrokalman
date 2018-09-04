from abc import ABC, abstractclassmethod

class AbstractKalman(ABC):

    @abstractclassmethod
    def predict(self):
        pass

    @abstractclassmethod
    def correct(self):
        pass

    def update(self):
        self.predict()
        self.correct()