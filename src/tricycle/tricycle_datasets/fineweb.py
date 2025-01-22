"""Prepares and manages web text data from Fineweb.

This module provides functionality to download, tokenize, and manage the
fineweb dataset. It includes utilities for data preparation and a custom
dataset class for efficient data loading.

Typical usage example:

  dataset = FineWeb(vocab_size=50257, split='train')
  tokens = dataset[0:1000]  # Get the first 1000 tokens
"""

import os
from collections import abc
from pathlib import Path
from typing import Literal

import numpy as np
import tiktoken
from tqdm.auto import tqdm

import datasets
from datasets import load_dataset

N_CORES = os.cpu_count()
SAVE_DIR = Path("datasets/fineweb")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
DTYPE = np.uint16


tokeniser = tiktoken.get_encoding("gpt2")


def tokenise_document(example):
    """Tokenizes a single document from the dataset.

    Args:
        example: A dictionary containing the 'text' field to be tokenized.

    Returns:
        A dictionary with 'ids' (tokenized text) and 'len' (number of tokens).
    """
    ids = tokeniser.encode_ordinary(
        example["text"]
    )  # encode_ordinary ignores any special tokens
    ids.append(tokeniser.eot_token)  # add the end of text token
    out = {"ids": ids, "len": len(ids)}
    return out


def prepare_data():
    """Downloads and tokenizes the coreparrot dataset.

    This function is adapted from Andrej Karpathy's NanoGPT:
    https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

    The function performs the following steps:
    1. Loads the dataset
    2. Splits it into train and validation sets
    3. Tokenizes the dataset
    4. Saves the tokenized data to binary files

    Note:
        This function uses OpenAI's tiktoken for tokenization due to
        performance considerations.
    """
    datasets.disable_caching()
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train"
    )
    split_dataset = dataset.train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["valid"] = split_dataset.pop(
        "test"
    )  # rename the test split to val

    # tokenise the dataset
    tokenised = split_dataset.map(
        tokenise_document,
        remove_columns=["text"],
        desc="Tokenising",
        num_proc=N_CORES,
    )

    # concatenate all the ids in each dataset into one large file we can use
    # for training
    for split, dset in tokenised.items():
        filename = SAVE_DIR / f"{split}.bin"

        n_tokens = np.sum(dset["len"])
        print(f"Found: {n_tokens} {split} tokens")

        arr = np.memmap(filename, dtype=DTYPE, mode="w+", shape=(n_tokens,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(
            range(total_batches), desc=f"writing {filename}"
        ):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


class FineWeb(abc.Sequence):
    """A custom dataset class for efficient loading of tokenized fineweb data.

    This class provides an interface to access tokenized fineweb data,
    supporting indexing and length operations. It also includes methods
    for encoding and decoding tokens.

    Attributes:
        vocab_size: An integer representing the vocabulary size.
        token_path: A Path object pointing to the tokenized data file.
        tokeniser_string: A string specifying the tokenizer to use (default: "gpt2").
        tokens: A numpy memmap of the tokenized data.

    Args:
        vocab_size: An integer specifying the vocabulary size.
        split: A string literal, either "train" or "valid", specifying the dataset split.
        token_path: An optional Path object for the tokenized data file.

    Raises:
        ValueError: If the tokenizer's max token value doesn't match the specified vocab size.
    """

    vocab_size: int
    token_path: Path
    tokeniser_string: str = "gpt2"
    tokens: np.ndarray

    def __init__(
        self,
        vocab_size: int,
        split: Literal["train"] | Literal["valid"],
        token_path: Path | None = None,
    ):
        self.vocab_size = vocab_size

        self.tokeniser = tiktoken.get_encoding(self.tokeniser_string)
        if self.tokeniser.max_token_value != vocab_size:
            raise ValueError(
                "Expected tokeniser.max_token_value == vocab_size. Found "
                f"{self.tokeniser.max_token_value=}, {vocab_size=}"
            )

        if token_path is None:
            self.token_path = SAVE_DIR / f"{split}.bin"
        else:
            self.token_path = token_path

        if not self.token_path.exists():
            prepare_data()

        assert self.token_path.exists()
        self.tokens = np.memmap(self.token_path, dtype=DTYPE, mode="r")

    def __getitem__(self, key):
        """Retrieves token(s) at the specified index or slice.

        Args:
            key: An integer index or slice object.

        Returns:
            The token(s) at the specified index or slice.
        """
        return self.tokens[key]

    def __len__(self):
        """Returns the total number of tokens in the dataset.

        Returns:
            An integer representing the number of tokens.
        """
        return len(self.tokens)

    def encode(self, *args):
        """Encodes the input text into tokens.

        Args:
            *args: Variable length argument list to be passed to the tokenizer.

        Returns:
            A list of integer token IDs.
        """
        return self.tokeniser.encode_ordinary(*args)

    def decode(self, *args):
        """Decodes the input tokens into text.

        Args:
            *args: Variable length argument list to be passed to the tokenizer.

        Returns:
            A string of decoded text.
        """
        return self.tokeniser.decode(*args)
