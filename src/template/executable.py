import importlib
from pathlib import Path
import sys
from template.trainer import Trainer, load_config_dict 
import json
from collections import namedtuple


def init_config_object(config_dict):
    # https://stackoverflow.com/a/34997118/17749529
    # https://stackoverflow.com/a/15882327/17749529
    # dictionary to object for dot access
    # namedtuple for immutability
    return json.loads(
        json.dumps(config_dict),
        object_hook=lambda d: namedtuple("Config", d.keys())(*d.values()),
    )

def prepare_data():
    file_name = sys.argv[2]
    path = Path(file_name)
    module = importlib.import_module(f"{__package__}.dataset.{path.stem}")
    module.prepare(path)


def train():
    config_file_path = Path(sys.argv[2]) / "config.toml"
    config_dict = load_config_dict(config_file_path)
    config = init_config_object(config_dict)
    experiment = importlib.import_module(
        "template.experiment." + config.experiment_name
    )
    data = experiment.Data(config)
    model = experiment.Model(config)
    t = Trainer(config, data, model)
    t.run()


def generate():
    config_file_path = Path(sys.argv[2]) / "config.toml"
    config_dict = load_config_dict(config_file_path)
    config = init_config_object(config_dict)
    experiment = importlib.import_module(
        "template.experiment." + config.experiment_name
    )
    data = experiment.Data(config)
    model = experiment.Model(config)
    t = Trainer(config, data, model)
    t.generate()


scripts = {"prepare_data": prepare_data, "train": train, 'generate':generate}


def main():
    script_name = sys.argv[1]
    scripts[script_name]()
