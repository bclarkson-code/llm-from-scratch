import time
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
from warnings import warn

import humanize
import numpy as np
from matplotlib import pyplot as plt

from tricycle import GPU_ENABLED, TRICYCLE_CONTEXT
from tricycle.configs import GPTConfig, SmolGPTConfig
from tricycle.exceptions import GPUDisabledException

if TYPE_CHECKING:
    from tricycle.models import GPT
    from tricycle.tensor import Tensor


class Dataset:
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

    def __next__(self):
        return self.__getitem__(next(iter(self)))


class UseMixedPrecision:
    def __init__(self):
        self.active = False

    def __enter__(self):
        self.active = True
        TRICYCLE_CONTEXT.use_mixed_precision = True

    def __exit__(self, *args, **kwargs):
        self.active = False
        TRICYCLE_CONTEXT.use_mixed_precision = False


def shapes_match(tensor_1: "Tensor", tensor_2: "Tensor") -> bool:
    """
    Binary operations can only be performed if the matrices are the same shape
    This function checks that we are allowed to apply a binary Op.
    """
    # sourcery skip: assign-if-exp, merge-duplicate-blocks, remove-redundant-if
    if tensor_1.is_batched and tensor_2.is_batched:
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape
    elif tensor_1.is_batched:
        shape_1 = tensor_1.shape[1:]
        shape_2 = tensor_2.shape
    elif tensor_2.is_batched:
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape[1:]
    else:
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape

    if shape_1 != shape_2:
        raise ValueError(
            f"Shapes {shape_1} and {shape_2} do not match: "
            f"{tensor_1.array.shape}, {tensor_2.array.shape}"
        )
    return shape_1 == shape_2


def smooth(iterable: Iterable, factor: float):
    """
    Use exponential smoothing to smooth an array
    """
    prev = 0
    for val in iterable:
        yield prev * factor + (val - prev) * factor
        prev = val


def r_squared(actual_values, predicted_values):
    """
    calculate R-squared metric.
    """
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)

    mean_actual = np.mean(actual_values)
    tss = np.sum((actual_values - mean_actual) ** 2)
    rss = np.sum((actual_values - predicted_values) ** 2)

    return 1 - (rss / tss)


def log_memory_and_time(stage: str, path: Path = Path("memory.log")):
    """
    Log the current GPU memory usage to a file
    """
    if not GPU_ENABLED:
        raise GPUDisabledException(
            "Cannot log GPU memory if GPU is not enabled"
        )

    import cupy

    if not path.exists():
        path.write_text(
            "stage,used_bytes_human,total_bytes_human,used_bytes,total_bytes,timestamp\n"  # noqa: E501
        )

    pool = cupy.get_default_memory_pool()
    now = time.perf_counter()

    used_bytes = humanize.naturalsize(pool.used_bytes())
    total_bytes = humanize.naturalsize(pool.total_bytes())
    with open(path, "a") as f:
        f.write(
            f"{stage},{used_bytes},{total_bytes},{pool.used_bytes()},{pool.total_bytes()},{now}\n"  # noqa: E501
        )


def optimal_n_tokens(model: "GPT", config: GPTConfig) -> tuple[int, int]:
    """
    Use chinchilla scaling to estimate the compute-optimal number of
    tokens to train on.
    See https://arxiv.org/abs/2404.10102 for details
    """
    # values from the appendix of the paper
    flops = [
        1.84e19,
        1.20e20,
        1.32e22,
        6.88e23,
        4.54e24,
        1.18e25,
        4.19e25,
        1.59e26,
        1.75e28,
    ]
    tokens = [
        7.7e9,
        20e9,
        219.5e9,
        1.7e12,
        4.3e12,
        7.1e12,
        13.4e12,
        26.5e12,
        292e12,
    ]

    # fit a linear regression
    slope, intercept = np.polyfit(np.log(flops), np.log(tokens), 1)

    n_parameters = sum(size for _, size, _ in model._contents())

    # rearrange regression to get number of tokens:
    #
    # assuming flops ~= 6 * n_tokens * n_parameters, we get
    # log(tokens) = slope * log(6 * n_tokens * n_parameters) + intercept
    # which rearranges to the following:
    power = 1 / (1 - slope)
    constant = (6**slope) * (n_parameters**slope) * np.exp(intercept)
    n_tokens = int(constant**power)

    tokens_per_step = (
        config.batch_size
        * config.gradient_accumulation_steps
        * config.context_window
    )
    tokens_per_parameter = n_tokens / n_parameters

    n_steps = n_tokens // tokens_per_step

    print("Chinchilla Optimal Parameters:")
    print(f" - Number of tokens: {humanize.intword(n_tokens)}")
    print(f" - Number of steps: {humanize.intword(n_steps)}")
    print(f" - Tokens per parameters: {tokens_per_parameter:.1f}")
    return n_tokens, n_steps


if __name__ == "__main__":
    config = SmolGPTConfig()
    optimal_n_tokens(config)
