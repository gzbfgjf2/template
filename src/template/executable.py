import importlib
from pathlib import Path
import sys
from template.trainer import Trainer, load_config_dict, init_config_object


def prepare_data():
    file_name = sys.argv[2]
    path = Path(file_name)
    module = importlib.import_module(f"{__package__}.dataset.{path.stem}")
    module.prepare(path)


def train():
    config_file_path = Path(sys.argv[2]) / "config.toml"
    config_dict = load_config_dict(config_file_path)
    config = init_config_object(config_dict)
    module = importlib.import_module(
        "template.experiment." + config.experiment_name
    )
    data = module.Data(config)
    model = module.Model(config)
    t = Trainer(config_dict, data, model)
    t.run()


scripts = {"prepare_data": prepare_data, "train": train}


def main():
    script_name = sys.argv[1]
    scripts[script_name]()
