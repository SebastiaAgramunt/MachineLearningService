from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class SalaryDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target
        assert len(features) == len(target)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
       return self.features[index], self.target[index]