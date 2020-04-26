from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    """
    An abstract class to train a model
    """
    def __init__(self, in_size: int, out_size: int):

        self.in_size = in_size
        self.out_size = out_size

    @abstractmethod
    def learn(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_local(self, path: str):
        pass
