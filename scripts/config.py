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
    def __init__(self, source_url: str, filename: str, data_dir: str, max_features: int, step_del_features: int, seed: int, test_size: float):
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
