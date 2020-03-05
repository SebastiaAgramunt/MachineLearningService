import torch 
import torch.nn as nn
from base import ModelTrainer
import pandas as pd

class BasicNet(ModelTrainer, torch.nn.Module):
    def __init__(self, in_size, out_size):

        super().__init__(in_size, out_size)
        torch.nn.Module.__init__(self)

        self.linear1 = nn.Linear(10, 20)
        self.act = nn.Sigmoid()

        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 1)

        for param in list(self.parameters()):
            torch.nn.init.normal_(param, std=0.1)

    def forward(self, x):
        x_ = self.linear1(x)
        x_ = self.act(x_)
        x_ = self.linear2(x_)
        x_ = self.act(x_)
        x_ = self.linear3(x_)
        return x_

    def predict(self, x, threshold=0.5):
        predictions = torch.sigmoid(self.forward(x))

        p = []
        for prediction in predictions:
            if prediction>threshold:
                p.append(1)
            else:
                p.append(0)
        return torch.tensor(p)

    def train(self):
        pass

    def save_local(self):
        pass


if __name__ == '__main__':
    a = BasicNet(10, 5)
    print(a.__dict__)

