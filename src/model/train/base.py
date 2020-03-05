from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    """
    An abstract class to train a recommender system model.
    ...
    Attributes
    ----------
    num_users : int
        number of total users
    num_items : int
        number of total items
    embedding_dim : int
        dimensionality of the embedding
    num_layers : int
        number of layers of te neural net
    dropout : float between 0, 1
        dropout probability
    Methods
    -------
    fit(self, *args, **kwargs)
        fit the model
    save_local(self, save: str) -> List[List[int]]
        save model to path save
    """
    def __init__(self,in_size: int, out_size: int):

        self.in_size = in_size
        self.out_size = out_size
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_local(self, path: str):
        pass