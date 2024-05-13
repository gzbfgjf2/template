import importlib
from pathlib import Path
import sys
from template.trainer import Trainer
import json
from collections import namedtuple
import torch
import tomllib


def get_checkpoint():
    experiment_path = Path(sys.argv[2])
    checkpoint_path = experiment_path / "checkpoint.ckpt"
    # why cpu
    # https://github.com/pytorch/pytorch/issues/7415#issuecomment-693424574
    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu")
        if checkpoint_path.exists()
        else None
    )
    return checkpoint


def load_config_dict(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config():
    config_file_path = Path(sys.argv[2]) / "config.toml"
    config_dict = load_config_dict(config_file_path)
    return init_config_object(config_dict)


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
    config = load_config()
    experiment = importlib.import_module(
        "template.experiment." + config.experiment_name
    )
    checkpoint = get_checkpoint()
    data = experiment.Data(config)
    model = experiment.Model(config, checkpoint)
    t = Trainer(config, data, model, checkpoint)
    t.run()


def generate():
    config = load_config()
    checkpoint = get_checkpoint()
    experiment = importlib.import_module(
        "template.experiment." + config.experiment_name
    )
    data = experiment.Data(config)
    model = experiment.Model(config, checkpoint)
    t = Trainer(config, data, model, checkpoint)
    t.generate()


scripts = {
    "prepare_data": prepare_data,
    "train": train,
    "generate": generate,
}


def main():
    script_name = sys.argv[1]
    scripts[script_name]()


if __name__ == "__main__":
    main()
