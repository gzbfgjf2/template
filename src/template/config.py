import sys
from pathlib import Path
import json
from collections import namedtuple
import tomllib


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
