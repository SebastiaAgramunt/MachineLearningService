import torch
import torch.nn as nn

from src.model.train.base import ModelTrainer
from src.dataloading.salary_loader import SalaryDataset


# TODO: implement cpu or gpu to try
class BasicNet(ModelTrainer, torch.nn.Module):
    def __init__(self, in_size=10, out_size=1):

        super().__init__(in_size, out_size)
        torch.nn.Module.__init__(self)

        self.linear1 = nn.Linear(in_size, 20)
        self.act = nn.Sigmoid()

        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, out_size)

        for param in list(self.parameters()):
            torch.nn.init.normal_(param, std=0.1)

    def forward(self, x):
        x_ = self.linear1(x)
        x_ = self.act(x_)
        x_ = self.linear2(x_)
        x_ = self.act(x_)
        x_ = self.linear3(x_)
        return x_

    def learn(self,
              data: SalaryDataset,
              learning_rate: float,
              epochs: int,
              batch_size=int
              ) -> None:

        data = torch.utils.data.DataLoader(data,
                                           batch_size=batch_size,
                                           shuffle=False)

        # Train mode (upddate weights)
        self.train()
        # Criterion binary cross entropy loss (we have two classes)
        criterion = nn.BCELoss()
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        losses = []
        for epoch in range(epochs):
            for d in data:
                x, y = d[0], d[1]
                # Clear the previous gradients
                optimizer.zero_grad()
                # Precit the output for Given input
                y_pred = torch.sigmoid(self.forward(x).float())
                # Calculate loss
                loss = criterion(y_pred, y.float())
                losses.append(loss.item())
                # Compute gradients
                loss.backward()
                # Adjust weights
                optimizer.step()
            if epoch % 5 == 0:
                print(f"Epoch:{epoch}, loss:{sum(losses)/len(losses)}")

    def save_local(self, path):
        torch.save(self, path)
