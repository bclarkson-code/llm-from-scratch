"""
This module prepares and handles the CodeParrot dataset: a dataset of python
files scraped from github

It downloads, tokenizes, and processes the CodeParrot dataset, creating memory-mapped
files for efficient data handling during training. The module also provides a
CodeParrot class for easy access to the processed data.

Typical usage example:

    dataset = CodeParrot(vocab_size=100000, split="train")
    tokens = dataset[0:1000]  # Get the first 1000 tokens
"""

import os
from collections import abc
from pathlib import Path
from typing import Literal

import numpy as np
import tiktoken
from tqdm.auto import tqdm

from datasets import load_dataset

N_CORES = os.cpu_count()
SAVE_DIR = Path("datasets/codeparrot")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
DTYPE = np.uint32

tokeniser = tiktoken.get_encoding("cl100k_base")


def tokenise_document(example):
    """
    Tokenizes a single document from the dataset.

    Args:
        example: A dictionary containing the document content.

    Returns:
        A dictionary with tokenized 'ids' and 'len' fields.
    """
    ids = tokeniser.encode_ordinary(
        example["content"]
    )  # encode_ordinary ignores any special tokens
    ids.append(tokeniser.eot_token)  # add the end of text token
    out = {"ids": ids, "len": len(ids)}
    return out


def prepare_data():
    """
    Downloads and tokenizes the CodeParrot dataset.

    This function splits the dataset into train and validation sets,
    tokenizes the content, and saves the tokenized data as memory-mapped files.

    Note:
        This script is adapted from Andrej Karpathy's NanoGPT:
        https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
    """
    split_dataset = dataset.train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["valid"] = split_dataset.pop(
        "test"
    )  # rename the test split to val

    # tokenise the dataset
    tokenised = split_dataset.map(
        tokenise_document,
        remove_columns=["content"],
        desc="Tokenising",
        num_proc=N_CORES,
    )

    # concatenate all the ids in each dataset into one large file we can use
    # for training
    for split, dset in tokenised.items():
        filename = SAVE_DIR / f"{split}.bin"

        n_tokens = np.sum(dset["len"], dtype=np.uint64)
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


class CodeParrot(abc.Sequence):
    """
    A class to handle the CodeParrot dataset.

    This class provides an interface to access the tokenized CodeParrot dataset,
    including methods for encoding and decoding text.

    Attributes:
        url: The source URL of the dataset.
        vocab_size: The size of the vocabulary.
        token_path: The path to the tokenized data file.
        tokeniser_string: The name of the tokenizer to use.
        tokens: The memory-mapped array of tokens.

    Args:
        vocab_size: The size of the vocabulary to use.
        split: The dataset split to use ("train" or "valid").
        token_path: Optional custom path to the tokenized data file.
    """

    url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # noqa: E501
    )
    vocab_size: int
    token_path: Path
    tokeniser_string: str = "cl100k_base"
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
        """
        Retrieves tokens at the specified index or slice.

        Args:
            key: An integer index or slice object.

        Returns:
            The token(s) at the specified index or slice.
        """
        return self.tokens[key]

    def __len__(self):
        """
        Returns the total number of tokens in the dataset.

        Returns:
            The length of the tokens array.
        """
        return len(self.tokens)

    def encode(self, *args):
        """
        Encodes the input text into tokens.

        Args:
            *args: The text to encode.

        Returns:
            A list of token ids.
        """
        return self.tokeniser.encode_ordinary(*args)

    def decode(self, *args):
        """
        Decodes the input tokens into text.

        Args:
            *args: The tokens to decode.

        Returns:
            The decoded text as a string.
        """
        return self.tokeniser.decode(*args)
