"""
avgm.datasets

This module contains classes and methods that help us deal with the data for this model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import h5py
import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from torch.utils import data
from typing_extensions import TypeAlias

Device: TypeAlias = Union[Literal["cuda"], Literal["gpu"]]


class DataCollection:
    """
    This class bundles together the tokenized data that was created by our tokenization
    process with the tokenizer used to create it. The data itself is stored in different
    datasets, according to our hold-out strategy for cross validation.

    Args:
        data_path (Path): Path to the H5 file that contains the data.
        tkn_path (Path): Path to the tokenizer vocabulary file.
        max_size (Optional[int]): Maximum sequence length for the input (Default: None)
        datasets (Tuple[str,...]): Tuple with which datasets we would like to load.
    """

    def __init__(
        self,
        data_path: Path,
        tkn_path: Path,
        max_size: Optional[int] = None,
        datasets: Tuple[str, ...] = ("train", "test", "valid"),
    ):
        self._load_tokenizer(tkn_path)
        self._read_hdf5(data_path, datasets=datasets, max_size=max_size)

    def _load_tokenizer(self, path: Path) -> None:
        """
        This method loads the tokenizer from its vocabulary file. The tokenizer used
        by this project is a Word Piece tokenizer similar to the one used by Bert.

        Args:
            path (Path): Path to the tokenizer vocabulary file.
        """
        self.UNK_TOKEN: str = "[UNK]"
        self.PAD_TOKEN: str = "[PAD]"
        self.tokenizer: models.WordPiece = Tokenizer(
            models.WordPiece.from_file(str(path), unk_token=self.UNK_TOKEN)
        )
        self.tokenizer.add_special_tokens([self.PAD_TOKEN, self.UNK_TOKEN])

        self.PAD_INDEX: int = self.tokenizer.token_to_id(self.PAD_TOKEN)
        self.UNK_INDEX: int = self.tokenizer.token_to_id(self.UNK_TOKEN)
        self.VOCAB_SIZE: int = self.tokenizer.get_vocab_size()

        self.tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=True,
            lowercase=True,
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")

    def _read_hdf5(
        self, path, datasets=("train", "valid", "test"), max_size: Optional[int] = None
    ):
        """
        This function read the data from the HDF5 file into the datasets, loading them
        into memory.

        Args:
            path (Path): Path to the HDF5 file.
            datasets (Tuple[str,...]): Tuple with which datasets should be loaded.
            max_size (Optional[int]): Maximum sequence length to load.
        """
        with h5py.File(path, "r") as f:
            self.datasets: Dict[str, Dataset] = {
                dataset: Dataset(f, dataset, max_size=max_size) for dataset in datasets
            }

    def __getitem__(self, item: str) -> Dataset:
        return self.datasets[item]

    def encode(self, text: str, use_ids=False) -> List[int]:
        """
        This function encode a string of text into the token IDs
        """
        tokenized = self.tokenizer.encode(text, add_special_tokens=False)
        return tokenized.ids if use_ids else tokenized.tokens

    def get_data_loader(
        self,
        dataset: str,
        batch_size: int = 256,
        device: Device = "cuda",
    ) -> data.DataLoader:
        """
        This method returns a Data Loader for a specific dataset within the data
        collection

        Args:
            dataset (str): The dataset which we want to retrieve.
            batch_size (int): The size of the batches that will be returned by the
                data loader on each iteration (Default: 256)
            device (Device): Which device should the data be loaded into (Default: cuda)
        """
        return data.DataLoader(
            self[dataset],
            batch_size,
            collate_fn=PadSequence(self.PAD_INDEX, device=device),
            shuffle=True,
        )


class PadSequence:
    """
    This class implements that collation function that will be used by the dataloaders
    to process each batch of data. This class has two main resposibilities:
        - Sort the batches of data by reversed sequence length (Required to training)
        - Pad the sequences that are smallers than the maximum size.

    Args:
        padding_value (int): Which value to be used as the padding element (Should be
            provider by the loader from the tokenizer)
        device (Device): Which device should the padded data be sent to (Default: cuda)
    """

    def __init__(self, padding_value: int, device: Device = "cuda"):
        self.padding_value: int = padding_value
        self.device: Device = device

    def __call__(self, batch: torch.tensor):
        """
        This method makes the object into a callable that will be called by the data
        loader.

        Args:
            batch (torch.tensor): Batch of data that will be processed.
        """
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sequences = [tokens.to(self.device) for tokens, _ in sorted_batch]
        lengths = torch.tensor(
            [len(tokens) for tokens in sequences], device=self.device, dtype=torch.int64
        )
        scores = torch.tensor(
            [scores for _, scores in sorted_batch],
            device=self.device,
            dtype=torch.int64,
        )
        sequences_padded = nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.padding_value
        )
        return (sequences_padded, lengths), scores


class Dataset(data.Dataset):
    """
    This class load the data from the HDF5 file and make it available for the data
    loader.

    Args:
        dataset (h5py.Dataset): HDF5 file that contains the data.
        dstype (str): Which partition (Train, Test, Valid) of the data to load.
        max_size (Optional[int]): Maximum size that will be enforced for sequences (
            Sequences longer than this value will be truncated). Default: No truncation.
    """

    def __init__(
        self, dataset: h5py.Dataset, dstype: str, max_size: Optional[int] = None
    ):
        super().__init__()
        self.tokens = [
            torch.tensor(item[slice(0, max_size)], dtype=torch.int64)
            for item in dataset[f"data/{dstype}/tokens"]
        ]
        self.scores = [
            torch.tensor([item], dtype=torch.int64)
            for item in dataset[f"data/{dstype}/scores"]
        ]
        assert len(self.tokens) == len(self.scores)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[item], self.scores[item]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return iter(zip(self.tokens, self.scores))

    def __len__(self) -> int:
        return len(self.tokens)
