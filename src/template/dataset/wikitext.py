from pathlib import Path
from tqdm import tqdm
import numpy as np
import tiktoken
from pathlib import Path
from datasets import DatasetDict, load_dataset, list_datasets
import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.text import Perplexity


def prepare(path: Path):
    wikitext = load_dataset("wikitext", "wikitext-103-v1")
    # for pyright
    assert isinstance(wikitext, DatasetDict)
    wikitext = wikitext.map(_tokenize, remove_columns=["text"])
    for split_name, dataset in wikitext.items():
        _prepare_split(path, split_name, dataset)


class Data:
    def __init__(self, config) -> None:
        self.config = config
        self.dataset_name = Path(__file__).parent.name
        data_folder = Path.cwd() / "data" / self.dataset_name
        np_map = lambda file_name: np.memmap(
            data_folder / file_name, dtype=np.uint16, mode="r"
        )
        self.train_data = np_map("train.bin")
        self.validation_data = np_map("validation.bin")
        self.test_data = np_map("test.bin")
        self.train = TorchDataset(config, self.train_data)
        self.validation = TorchDataset(config, self.validation_data)
        self.test = TorchDataset(config, self.test_data)
        self.encoding = tiktoken.get_encoding("gpt2")

    def train_loader(self):
        return DataLoader(
            self.train, batch_size=10, shuffle=False, pin_memory=True
        )

    def valuation_loader(self):
        return DataLoader(
            self.validation, batch_size=10, shuffle=False, pin_memory=True
        )

    def compute_metric(self, predictions, labels):
        metric = Perplexity(device=predictions[0].device)
        assert len(predictions) == len(labels)
        for prediction, label in zip(predictions, labels):
            metric.update(prediction, label)
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
        x, y = np.split(data, [-1])
        return x, y


def _tokenize(example):
    encoding = tiktoken.get_encoding("gpt2")
    ids = encoding.encode_ordinary(example["text"])
    ids.append(encoding.eot_token)
    out = {"ids": ids, "len": len(ids)}
    return out


def _prepare_split(path, split_name, dataset):
    arr_len = np.sum(dataset["len"], dtype=np.uint64)
    filename = path / f"{split_name}.bin"
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    total_batches = 1024
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        batch = dataset.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
