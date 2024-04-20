from pathlib import Path
import requests
import os
import numpy as np
import pickle
import tiktoken
from torch.utils.data import DataLoader, Dataset
import torch
from torcheval.metrics.text import Perplexity


def prepare(path: Path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(path / "input.text", "w") as f:
        f.write(requests.get(data_url).text)
    text = requests.get(data_url).text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [
            stoi[c] for c in s
        ]  # encoder: take a string, output a list of integers

    def decode(l):
        return "".join(
            [itos[i] for i in l]
        )  # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(text)
    train_data = text[: int(n * 0.9)]
    val_data = text[int(n * 0.9) :]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(path / "train.bin")
    val_ids.tofile(path / "validation.bin")

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(path / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # length of dataset in characters:  1115394
    # all the unique characters:
    #  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    # vocab size: 65
    # train has 1003854 tokens
    # val has 111540 tokens


class Data:
    def __init__(self, config) -> None:
        self.config = config
        self.dataset_name = Path(__file__).stem
        data_folder = Path.cwd() / "data" / self.dataset_name
        print(data_folder)
        np_map = lambda file_name: np.memmap(
            data_folder / file_name, dtype=np.uint16, mode="r"
        )
        self.train_data = np_map("train.bin")
        self.validation_data = np_map("validation.bin")
        self.test_data = np_map("validation.bin")
        self.train = TorchDataset(config, self.train_data)
        self.validation = TorchDataset(config, self.validation_data)
        self.test = TorchDataset(config, self.test_data)
        self.encoding = tiktoken.get_encoding("gpt2")

    def train_loader(self):
        return DataLoader(
            self.train, batch_size=10, shuffle=False, pin_memory=True
        )

    def validation_loader(self):
        return DataLoader(
            self.validation, batch_size=10, shuffle=False, pin_memory=True
        )

    def compute_metric(self, predictions, labels):
        metric = Perplexity(device=predictions[0].device)
        for prediction, label in zip(predictions, labels):
            metric.update(prediction, label[1])
        return {"perplexity": metric.compute()}


class TorchDataset(Dataset):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.token_size = len(self.data)
        self.sequence_length = self.config.sequence_length
        # avoid overflow
        self.len = self.token_size // self.sequence_length - 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        data = self.data[start : start + self.sequence_length]
        data = torch.from_numpy(data.astype(np.int64))
        x = data[:-1]
        y = data[1:]
        return x, y
