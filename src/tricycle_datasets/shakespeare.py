"""Provides classes for handling Shakespeare datasets.

This module contains two main classes:
1. Shakespeare: For handling tokenized Shakespeare text using BPE tokenization.
2. ShakespeareChar: For handling character-level Shakespeare text.

Both classes provide methods for downloading, tokenizing, encoding, and decoding
Shakespeare's text.

Typical usage example:

  shakespeare = Shakespeare(1024)
  char_shakespeare = ShakespeareChar()
"""

import pickle
from collections import abc
from pathlib import Path

import numpy as np
import requests

from tricycle.tokeniser import BPETokeniser


class Shakespeare(abc.Sequence):
    """A class for handling tokenized Shakespeare text using BPE tokenization.

    This class downloads the Shakespeare dataset, tokenizes it using BPE,
    and provides methods for encoding and decoding text.

    Attributes:
        url: A string containing the URL for the Shakespeare dataset.
        vocab_size: An integer representing the size of the vocabulary.
        token_path: A Path object for the tokenized data file.
        raw_data_path: A Path object for the raw data file.
        tokens: A numpy array containing the tokenized data.
        tokeniser: A BPETokeniser object for tokenization.
    """

    url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # noqa: E501
    )
    vocab_size: int
    token_path: Path
    raw_data_path: Path
    tokens: np.ndarray

    def __init__(
        self,
        vocab_size: int,
        token_path: Path | None = None,
        raw_data_path: Path = Path("datasets/shakespeare/raw_data.txt"),
        tokeniser_path: Path = Path("datasets/shakespeare/tokeniser.pkl"),
    ):
        """Initializes the Shakespeare object.

        Args:
            vocab_size: An integer representing the size of the vocabulary.
            token_path: A Path object for the tokenized data file. If None, a default path is used.
            raw_data_path: A Path object for the raw data file.
            tokeniser_path: A Path object for the tokeniser pickle file.
        """
        if token_path is None:
            token_path = Path(f"datasets/shakespeare/tokens_{vocab_size}.pkl")

        self.vocab_size = vocab_size
        self.raw_data_path = raw_data_path
        self.token_path = token_path
        self.tokeniser_path = tokeniser_path

        if self.tokeniser_path.exists():
            with open(self.tokeniser_path, "rb") as f:
                self.tokeniser = pickle.load(f)
        else:
            self.tokeniser = None

        if not self.token_path.exists():
            self.tokeniser = self.generate()
            self.tokeniser_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tokeniser_path, "wb") as f:
                pickle.dump(self.tokeniser, f)

            self.tokens = self.tokeniser.tokens
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_path, "wb") as f:
                pickle.dump(self.tokens, f)
        else:
            with open(self.token_path, "rb") as f:
                self.tokens = pickle.load(f)

    def download(self):
        """Downloads the Shakespeare dataset.

        The downloaded data is saved to the path specified by raw_data_path.
        """
        raw_data = requests.get(self.url).text
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.raw_data_path, "w") as f:
            f.write(raw_data)

    def generate(self) -> BPETokeniser:
        """Downloads and tokenizes the Shakespeare dataset.

        Returns:
            A BPETokeniser object trained on the Shakespeare dataset.
        """
        self.download()
        raw_data = np.array(
            list(self.raw_data_path.read_bytes()), dtype=np.int32
        )
        if self.tokeniser is None:
            self.tokeniser = BPETokeniser(self.vocab_size)
        return self.tokeniser.train_ints(raw_data, loading_bar=True)

    def __getitem__(self, idx: int) -> int | list[int]:
        """Returns the token(s) at the specified index.

        Args:
            idx: An integer index or slice.

        Returns:
            The token(s) at the specified index.
        """
        return self.tokens[idx]

    def __len__(self) -> int:
        """Returns the number of tokens in the dataset.

        Returns:
            An integer representing the number of tokens.
        """
        return len(self.tokens)

    def encode(self, *args):
        """Encodes the input using the BPE tokenizer.

        Args:
            *args: Arguments to pass to the tokenizer's encode method.

        Returns:
            The encoded input.
        """
        return self.tokeniser.encode(*args)

    def decode(self, *args):
        """Decodes the input using the BPE tokenizer.

        Args:
            *args: Arguments to pass to the tokenizer's decode method.

        Returns:
            The decoded input.
        """
        return self.tokeniser.decode(*args)


class ShakespeareChar(abc.Sequence):
    """A class for handling character-level Shakespeare text.

    This class downloads the Shakespeare dataset and provides methods for
    encoding and decoding text at the character level.

    Attributes:
        url: A string containing the URL for the Shakespeare dataset.
        vocab_size: An integer representing the size of the vocabulary.
        raw_data_path: A Path object for the raw data file.
        chars: A list of integers representing the characters in the dataset.
    """

    url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # noqa: E501
    )
    vocab_size: int
    raw_data_path: Path
    chars: list[int]

    def __init__(
        self,
        raw_data_path: Path = Path("datasets/shakespeare/raw_data.txt"),
    ):
        """Initializes the ShakespeareChar object.

        Args:
            raw_data_path: A Path object for the raw data file.
        """
        self.raw_data_path = raw_data_path
        self.chars = self.generate()
        self.vocab_size = len(set(self.chars))

    def encode(self, chars: list[int] | str):
        """Encodes the input characters into character IDs.

        Args:
            chars: A list of integers or a string to encode.

        Returns:
            A list of integer character IDs.
        """
        if isinstance(chars, str):
            chars = [ord(i) for i in chars]
        return [self.char_ids[c] for c in chars]

    def decode(self, char_ids: list[int]):
        """Decodes the input character IDs into characters.

        Args:
            char_ids: A list of integer character IDs to decode.

        Returns:
            A list of decoded characters.
        """
        inv_char_ids = {i: c for c, i in self.char_ids.items()}
        return [inv_char_ids[i] for i in char_ids]

    def download(self):
        """Downloads the Shakespeare dataset.

        The downloaded data is saved to the path specified by raw_data_path.
        """
        raw_data = requests.get(self.url).text
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.raw_data_path, "w") as f:
            f.write(raw_data)

    def generate(self) -> list[int]:
        """Downloads and processes the Shakespeare dataset.

        Returns:
            A list of integers representing the characters in the dataset.
        """
        if not self.raw_data_path.exists():
            self.download()

        raw_data = list(self.raw_data_path.read_bytes())
        self.char_ids = {c: i for i, c in enumerate(set(raw_data))}
        return self.encode(raw_data)

    def __getitem__(self, idx: int) -> int | list[int]:
        """Returns the character(s) at the specified index.

        Args:
            idx: An integer index or slice.

        Returns:
            The character(s) at the specified index.
        """
        return self.chars[idx]

    def __len__(self) -> int:
        """Returns the number of characters in the dataset.

        Returns:
            An integer representing the number of characters.
        """
        return len(self.chars)
