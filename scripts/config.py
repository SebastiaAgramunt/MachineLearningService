import os, sys
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

# to use in {root_path} at init file
def project_abs_path():
    return Path(__file__).parent.parent

def create_dirs(*paths):
    for path in paths:
        path = path.split('/')
        path = '/'.join(path[:-1])

        if not os.path.exists(path):
            os.mkdir(path)

this_filepath = os.path.dirname(os.path.abspath(__file__))
config_path = this_filepath + '/config.ini'

config_preproc = ConfigParser(interpolation=ExtendedInterpolation())
config_preproc.read(config_path)

root_path = str(project_abs_path())

class ParseDataPreprocessing:
    def __init__(self, source_url: str, filename: str, data_dir: str,
     max_features: int, step_del_features: int, seed: int, test_size: float):
        self.source_url = source_url
        self.filename = filename
        self.data_dir = data_dir
        self.max_features = max_features
        self.step_del_features = step_del_features
        self.seed = seed
        self.test_size = test_size

        assert self.test_size>0, "test size must be larger than 0"
        assert self.test_size<1, "test size must be smaller than 1"

    @classmethod
    def build_from_config(cls, config: ConfigParser = config_preproc):
        source_url = config.get('preprocessing', 'SOURCE_URL')
        filename = config.get('preprocessing', 'FILENAME')
        data_dir = config.get('preprocessing', 'DATA_DIR').replace('{root_path}', root_path)
        max_features = config.getint('preprocessing', 'MAX_FEATURES')
        step_del_features = config.getint('preprocessing', 'STEP_DEL_FEATURES')
        seed = config.getint('preprocessing', 'SEED')
        test_size = config.getfloat('preprocessing', 'TEST_SIZE')

        return cls(source_url, filename, data_dir, max_features, 
            step_del_features, seed, test_size)

class ParseTraining:
    def __init__(self, data_dir: str, model_path: str, model_name: str,
        learning_rate: float, epochs: int, batch_size: int):
        self.data_dir = data_dir
        self.model_path = model_path
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    @classmethod
    def build_from_config(cls, config: ConfigParser = config_preproc):
        data_dir = config.get('preprocessing', 'DATA_DIR').replace('{root_path}', root_path)
        model_path = config.get('training', 'MODEL_PATH').replace('{root_path}', root_path)
        model_name = config.get('training', 'MODEL_NAME')

        learning_rate = config.getfloat('training', 'LEARNING_RATE')
        epochs = config.getint('training', 'EPOCHS')
        batch_size = config.getint('training', 'BATCH_SIZE')
        
        return cls(data_dir, model_path, model_name, learning_rate, epochs, 
            batch_size)

class ParseEvaluation:
    def __init__(self, model_path: str, model_name: str, data_dir: str,
     test_log: str):
        self.model_path = model_path
        self.model_name = model_name
        self.data_dir = data_dir
        self.test_log = test_log

    @classmethod
    def build_from_config(cls, config: ConfigParser = config_preproc):
        model_path = config.get('training', 'MODEL_PATH').replace('{root_path}', root_path)
        model_name = config.get('training', 'MODEL_NAME')
        data_dir = config.get('preprocessing', 'DATA_DIR').replace('{root_path}', root_path)
        test_log = config.get('testing', 'TEST_LOG')
        
        return cls(model_path, model_name, data_dir, test_log)