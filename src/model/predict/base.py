from abc import ABC, abstractmethod


class ModelPredictor(ABC):
    """
    An abstract class to predict in a model
    """
    def __init__(self, filepath):
        self.filepath = filepath
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass