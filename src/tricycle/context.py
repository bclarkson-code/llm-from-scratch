"""Defines the context for Tricycle operations.

This module provides a dataclass for storing Tricycle context information,
including mixed precision and loss scaling settings.
"""

from dataclasses import dataclass


@dataclass
class TricycleContext:
    """A dataclass to store Tricycle context information.

    Attributes:
        use_mixed_precision (bool): Flag to enable mixed precision. Default is False.
            Note: It's recommended to use the tricycle/utils.py:UseMixedPrecision
            context manager for mixed precision training instead of modifying this directly.

        loss_scale_factor (int): Factor to scale the loss when using mixed precision.
            This helps prevent under and overflowing. Default is 128.
    """

    use_mixed_precision: bool = False
    loss_scale_factor: int = 128


# Global instance of TricycleContext
TRICYCLE_CONTEXT = TricycleContext()
