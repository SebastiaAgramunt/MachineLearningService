import sys
import os.path
from pathlib import Path

# to add above path so that we can import built libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
 os.path.pardir)))

from src.model.train.basicnet import BasicNet
from src.dataloading.salary_loader import SalaryDataset
from config import ParseTraining

def create_dirs(*paths):
    for path in paths:
        path = path.split('/')
        path = '/'.join(path[:-1])

        if not os.path.exists(path):
            os.mkdir(path)


if __name__ == '__main__':

    config = ParseTraining.build_from_config()

    model = BasicNet()
    data = SalaryDataset.build_from_files(config.data_dir+'/processed/'+\
        "X_train_norm.csv", config.data_dir+'/processed/'+"y_train_norm.csv")

    model.learn(data=data, learning_rate=config.learning_rate,
     epochs=config.epochs, batch_size=config.batch_size)

    create_dirs(config.model_path+'/'+config.model_name)
    model.save_local(config.model_path+'/'+config.model_name)



