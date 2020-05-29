import os
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


class ParseServing:
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name

    @classmethod
    def build_from_config(cls, config: ConfigParser = config_preproc):
        model_path = config.get('serve', 'MODEL_PATH').replace(
            '{root_path}', root_path)
        model_name = config.get('serve', 'MODEL_NAME')

        return cls(model_path, model_name)

