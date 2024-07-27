"""
Tricycle: A deep learning framework.

This module initializes the Tricycle framework and imports its various components.
It also checks for GPU support using CuPY.

Attributes:
    GPU_ENABLED (bool): Indicates whether GPU support is available.

Imports:
    Various submodules of the Tricycle framework.
"""

from warnings import warn

try:
    import cupy

    # check that we can create a cupy array and operate on it
    cupy.array([1, 2, 3]) * 2
    GPU_ENABLED = True
except ImportError:
    warn("Could not find CuPY, disabling GPU features")
    GPU_ENABLED = False
except Exception as e:
    GPU_ENABLED = False
    warn(f"Failed to build cupy array: {e}. Disabling GPU features")

from . import (
    activation,
    attention,
    binary,
    blocks,
    configs,
    context,
    dataset,
    einsum,
    exceptions,
    functions,
    initialisers,
    layers,
    loss,
    models,
    ops,
    optimisers,
    reduce,
    scheduler,
    tensor,
    tokeniser,
    unary,
    utils,
    weakset,
)

__all__ = [
    "activation",
    "attention",
    "binary",
    "blocks",
    "configs",
    "context",
    "dataset",
    "einsum",
    "exceptions",
    "functions",
    "initialisers",
    "layers",
    "loss",
    "models",
    "ops",
    "optimisers",
    "reduce",
    "scheduler",
    "tensor",
    "tokeniser",
    "unary",
    "utils",
    "weakset",
]
