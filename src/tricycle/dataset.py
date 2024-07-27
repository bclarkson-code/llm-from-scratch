import random
from typing import Sequence

import numpy as np

from tricycle.tensor import Tensor


class Dataset:
    """
    An in-memory dataset: not suitable for large datasets.

    This class represents a basic dataset with inputs and corresponding outputs.
    It supports iteration, indexing, and shuffling of data.

    Attributes:
        inputs: A sequence of input data.
        outputs: A sequence of output data corresponding to the inputs.
        _indices: A list of indices for accessing data.
        _index: The current index for iteration.

    Args:
        inputs: A sequence of input data.
        outputs: A sequence of output data.

    Raises:
        AssertionError: If the length of inputs and outputs are not equal.
    """

    def __init__(self, inputs: Sequence, outputs: Sequence):
        assert len(inputs) == len(outputs)
        self.inputs = inputs
        self.outputs = outputs
        self._indices = list(range(len(inputs)))
        self._index = 0

    def __iter__(self):
        """Returns the dataset object as an iterator."""
        return self

    def __next__(self):
        """
        Returns the next item in the dataset.

        Raises:
            StopIteration: If all items have been iterated over.
        """
        if self._index >= len(self.inputs):
            raise StopIteration

        result = self[self._index]
        self._index += 1
        return result

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx: int):
        """
        Returns the item at the specified index.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A tuple containing the input and output at the specified index.
        """
        idx = self._indices[idx]
        return self.inputs[idx], self.outputs[idx]

    def shuffle(self):
        """
        Shuffles the dataset indices.

        Returns:
            The dataset object with shuffled indices.
        """
        np.random.shuffle(self._indices)
        return self

    def to_tensor(self):
        """
        Converts inputs and outputs to Tensor objects.

        Returns:
            The dataset object with inputs and outputs as Tensors.
        """
        self.inputs = [Tensor(x) for x in self.inputs]
        self.outputs = [Tensor(x) for x in self.outputs]
        return self

    def reset(self):
        """
        Resets the iteration index to 0.

        Returns:
            The dataset object with reset index.
        """
        self._index = 0
        return self

    def copy(self):
        """
        Creates a shallow copy of the dataset.

        Returns:
            A new Dataset object with copied inputs and outputs.
        """
        return Dataset(self.inputs.copy(), self.outputs.copy())


class InfiniteBatchDataset(Dataset):
    """
    An infinite batch dataset that generates random batches.

    This class extends the Dataset class to provide infinite batches of data.
    It randomly selects items from the dataset to form batches.

    Attributes:
        is_infinite: A boolean indicating if the dataset is infinite.
        _to_tensor: A boolean indicating if the data should be converted to tensors.
        is_batched: A boolean indicating if the data is batched.
        batch_size: The size of each batch.

    Args:
        inputs: A sequence of input data.
        outputs: A sequence of output data.
        batch_size: The size of each batch.
    """

    is_infinite = True
    _to_tensor = False
    is_batched = True

    def __init__(self, inputs: Sequence, outputs: Sequence, batch_size: int):
        super().__init__(inputs, outputs)
        self.batch_size = batch_size

    def __next__(self):
        """Returns the next batch of items."""
        result = self[self._index]
        self._index += 1
        return result

    def __len__(self):
        """Returns -1 to indicate an infinite dataset."""
        return -1

    def __getitem__(self, idx: int):
        """
        Returns a randomly generated batch of items.

        Args:
            idx: The index used as a seed for random generation.

        Returns:
            A tuple containing batches of inputs and outputs.
        """
        random.seed(idx)
        indices = [
            random.randint(0, len(self.inputs) - 1)
            for _ in range(self.batch_size)
        ]
        batch_inputs = np.vstack([self.inputs[i] for i in indices])
        batch_outputs = np.vstack([self.outputs[i] for i in indices])

        if self._to_tensor:
            batch_inputs = Tensor(
                batch_inputs,
                is_batched=self.is_batched,
                dtype=batch_outputs.dtype,
            )
            batch_outputs = Tensor(
                batch_outputs,
                is_batched=self.is_batched,
                dtype=batch_outputs.dtype,
            )
        return batch_inputs, batch_outputs

    def to_tensor(self):
        """
        Sets the flag to convert data to tensors.

        Returns:
            The dataset object with _to_tensor flag set to True.
        """
        self._to_tensor = True
        return self


class CausalLMDataset:
    """
    A dataset for causal language modeling tasks.

    This class provides functionality for creating batches of token sequences
    for training causal language models.

    Attributes:
        tokens: The input token sequence.
        vocab_size: The size of the vocabulary.
        batch_size: The size of each batch.
        context_window: The size of the context window.
        is_batch: A boolean indicating if the data is batched.
        as_tensor: A boolean indicating if the data should be returned as tensors.
        _idx: The current index for iteration.
        batch_indices: The indices used for batching.
        should_one_hot_encode: A boolean indicating if outputs should be one-hot encoded.
        device: The device (GPU) to use for tensors.

    Args:
        tokens: The input token sequence.
        vocab_size: The size of the vocabulary.
        batch_size: The size of each batch.
        context_window: The size of the context window.
        should_one_hot_encode: Whether to one-hot encode the outputs.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        vocab_size: int,
        batch_size: int,
        context_window: int,
        should_one_hot_encode: bool = False,
    ):
        self.tokens = tokens
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.context_window = context_window
        self.is_batch = False
        self.as_tensor = False
        self._idx = 0
        self.batch_indices = None
        self.should_one_hot_encode = should_one_hot_encode
        self.device = None

    def __len__(self):
        """Returns the length of the dataset based on batching configuration."""
        return (
            (len(self.tokens) - self.context_window - self.batch_size - 1)
            // self.batch_size
            if self.is_batch
            else len(self.tokens) - 1
        )

    def __getitem__(self, idx: int):
        """
        Returns a batch or single item from the dataset.

        Args:
            idx: The index of the item or batch to retrieve.

        Returns:
            A tuple containing input and output sequences.
        """
        if self.is_batch:
            start = idx * self.batch_size
            end = (idx + 1) * self.batch_size
            indices = self.batch_indices[start:end]
            batches = [
                self.tokens[i : i + self.context_window + 1] for i in indices
            ]
            inputs = np.vstack([b[:-1] for b in batches])
            outputs = np.vstack([b[1:] for b in batches])
        else:
            start = idx * self.context_window
            end = (idx + 1) * self.context_window + 1
            tokens = self.tokens[start:end]
            inputs = tokens[:-1]
            outputs = tokens[1:]

        if self.as_tensor:
            inputs = Tensor(
                inputs,
                requires_grad=False,
                name="inputs",
                is_batched=self.is_batch,
                dtype=outputs.dtype,
            )
            outputs = Tensor(
                outputs,
                requires_grad=False,
                name="output",
                is_batched=self.is_batch,
                dtype=outputs.dtype,
            )
            if self.device is not None:
                inputs.to_gpu(self.device)
                outputs.to_gpu(self.device)

        return inputs, outputs

    def __iter__(self):
        """Returns the dataset object as an iterator."""
        self._idx = 0
        return self

    def __next__(self):
        """
        Returns the next item or batch in the dataset.

        Raises:
            StopIteration: If all items have been iterated over.
        """
        if self._idx >= len(self):
            raise StopIteration

        result = self[self._idx]
        self._idx += 1
        return result

    def batch(self):
        """
        Configures the dataset for batch processing.

        Returns:
            The dataset object configured for batch processing.
        """
        print("batching")
        self.is_batch = True
        self.batch_indices = np.arange(
            len(self.tokens) - self.context_window - 1
        )
        return self

    def unbatch(self):
        """
        Configures the dataset for non-batch processing.

        Returns:
            The dataset object configured for non-batch processing.
        """
        self.is_batch = False
        return self

    def shuffle(self):
        """
        Shuffles the batch indices.

        Returns:
            The dataset object with shuffled batch indices.

        Raises:
            NotImplementedError: If trying to shuffle a non-batched dataset.
        """
        print("shuffling")
        if not self.is_batch and self.batch_indices is not None:
            raise NotImplementedError(
                "Shuffling non-batched datasets is not currently supported"
            )
        else:
            n_batches = len(self.tokens) - self.context_window - 1
            self.batch_indices = np.random.choice(
                n_batches, size=n_batches, replace=False
            )
        return self

    def to_gpu(self, device: int = 0):
        """
        Sets the device for GPU processing.

        Args:
            device: The GPU device number.

        Returns:
            The dataset object configured for GPU processing.
        """
        self.device = device
        return self

    def from_gpu(self):
        """
        Resets the device to CPU processing.

        Returns:
            The dataset object configured for CPU processing.
        """
        self.device = None
        return self

    def to_tensor(self):
        """
        Configures the dataset to return tensors.

        Returns:
            The dataset object configured to return tensors.
        """
        print("converting to tensor")
        self.as_tensor = True
        return self
