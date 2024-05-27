import importlib
from pathlib import Path
import sys
import torch
from template.trainer import Trainer
from template.config import load_config
from template.ddp import DdpManager


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
    ddp = DdpManager(config)
    checkpoint = get_checkpoint()
    data = experiment.Data(config)
    model = experiment.Model(config, checkpoint)
    t = Trainer(config, data, model, ddp, checkpoint)
    t.run()
    ddp.destroy()


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
