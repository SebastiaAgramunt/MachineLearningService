import torch
import pandas as pd

from src.model.predict.base import ModelPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
# from src.dataloading.salary_loader import SalaryDataset


def evaluate_classification(y, y_hat, y_proba):
    return {
            "Accuracy": accuracy_score(y, y_hat),
            "AUC": roc_auc_score(y, y_proba),
            "Precision": precision_score(y, y_hat),
            "Recall": recall_score(y, y_hat),
            "F1-score": f1_score(y, y_hat)
            }


# TODO: implement cpu or gpu to try
class BasicNetPredict(ModelPredictor, torch.nn.Module):
    def __init__(self, filepath: str):

        super().__init__(filepath)
        torch.nn.Module.__init__(self)
        self.model = torch.load(filepath)
        self.model.eval()

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x, threshold=0.5):
        predictions = self.forward(x)
        print(predictions[2])
        p = []
        for prediction in predictions:
            if prediction > threshold:
                p.append(1)
            else:
                p.append(0)
        return torch.tensor(p)

    def test(self, inputs: pd.DataFrame, targets: pd.DataFrame):
        x = torch.from_numpy(inputs.values).float().to("cpu")

        y = torch.tensor(targets.values,
                         dtype=torch.long,
                         device="cpu").reshape(-1, 1)

        # self.model.eval()
        torch.set_printoptions(edgeitems=3)
        y_hat = self.predict(x)
        y_proba = self.forward(x).detach().numpy()

        return evaluate_classification(y.numpy(), y_hat, y_proba)


if __name__ == '__main__':
    pass
