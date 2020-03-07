from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import torch 

# TODO: Add device not 'cup' by default
class SalaryDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target
        assert len(features) == len(target)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
       return self.features[index], self.target[index]


    @classmethod
    def build_from_files(cls, predictors_filepath: str, target_filepath: str):

        inputs = pd.read_csv(predictors_filepath)
        targets = pd.read_csv(target_filepath)

        X = torch.from_numpy(inputs.values).float().to("cpu")
        y = torch.tensor(targets.values, dtype=torch.long, device="cpu")\
        .reshape(-1, 1)

        return cls(X, y)



