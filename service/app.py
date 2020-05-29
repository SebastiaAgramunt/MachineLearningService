import sys
import os.path
from pathlib import Path
from create_service import create_service

# to add above path so that we can import built libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                os.path.pardir)))

from src.model.predict.basicpredict import BasicNetPredict
from config import ParseServing


config = ParseServing.build_from_config()
model = BasicNetPredict(config.model_path + '/' + config.model_name)

# the final app served
app = create_service(model)