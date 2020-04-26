import sys
import os.path
import pandas as pd
import json


# to add above path so that we can import built libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                os.path.pardir)))

from src.model.predict.basicpredict import BasicNetPredict
from config import ParseEvaluation


if __name__ == '__main__':
    config = ParseEvaluation.build_from_config()

    # load stored model
    model = BasicNetPredict(config.model_path + '/' + config.model_name)

    # load data for test
    inputs = pd.read_csv(config.data_dir + '/processed/' + "X_test_norm.csv",
                         header=None)
    targets = pd.read_csv(config.data_dir + '/processed/' + "y_test_norm.csv",
                          header=None)

    # test on model
    test_data = model.test(inputs, targets)

    # save results
    with open(config.model_path + '/' + config.test_log, "w+") as file:
        file.write(json.dumps(test_data)+'\n')

    print(test_data)
