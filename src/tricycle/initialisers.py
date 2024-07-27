"""Initializers module for tensor initialization.

This module provides functions for initializing tensors with specific
distributions or patterns.
"""

import numpy as np

from tricycle.ops import Tensor
from tricycle.tensor import DEFAULT_DTYPE


def init_xavier(shape: tuple[int, int], name: str = "") -> Tensor:
    """Initialize a tensor with Xavier/Glorot initialization.

    This function implements Xavier/Glorot initialization, which helps in
    setting initial random weights for neural networks. It's particularly
    useful for maintaining the scale of gradients across layers.

    Args:
        shape: A tuple of two integers (f_in, f_out), where f_in is the number
            of input units and f_out is the number of output units.
        name: An optional string to name the created tensor. Defaults to an
            empty string.

    Returns:
        A Tensor object initialized with Xavier/Glorot initialization.

    Raises:
        ValueError: If the shape tuple does not contain exactly two integers.

    Example:
        >>> weight = init_xavier((100, 50), name="layer1_weights")
    """
    f_in, f_out = shape
    bound = np.sqrt(6) / np.sqrt(f_in + f_out)
    out = Tensor(
        np.random.uniform(low=-bound, high=bound, size=shape),
        dtype=DEFAULT_DTYPE,
        name=name,
    )
    return out
