import logging

import numpy as np

from tricycle.context import TRICYCLE_CONTEXT
from tricycle.layers import Dense, Layer
from tricycle.loss import MeanSquaredError
from tricycle.optimisers import StochasticGradientDescent
from tricycle.tensor import Tensor
from tricycle.utils import UseMixedPrecision

logger = logging.getLogger(__name__)


class LongBoi(Layer):
    """
    A very deep MLP with no nonlinearities, designed to underflow in mixed
    precision training
    """

    def __init__(self, n_layers: int = 16):
        self.layers = [
            Dense(to_size=16, from_size=16, name=f"layer_{i}")
            for i in range(n_layers)
        ]

    def forward(self, tensor: Tensor) -> Tensor:
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def update(self, optimiser):
        for layer in self.layers:
            layer.update(optimiser)


def test_can_train_in_mixed_precision():
    """
    Check that a model can be trained in mixed precision without overflowing

    We're using a very deep model with no nonlinearities that should cause
    gradient issues if mixed precision is broken
    """
    np.random.seed(0)
    learning_rate = 1e-3
    weight_decay = 1e-1
    model = LongBoi(64)

    loss_fn = MeanSquaredError()
    optimiser = StochasticGradientDescent(
        learning_rate=learning_rate, weight_decay=weight_decay, logger=logger
    )

    inputs = Tensor(
        np.random.random(
            (32, 16),
        ),
        is_batched=True,
        requires_grad=False,
    )
    outputs = Tensor(
        np.random.random(
            (32, 16),
        ),
        is_batched=True,
        requires_grad=False,
    )

    with UseMixedPrecision():
        first_loop = True
        for step in range(100):
            logits = model(inputs)
            loss = loss_fn(outputs, logits)
            loss.backward()
            loss = loss.numpy().item() / TRICYCLE_CONTEXT.loss_scale_factor
            if first_loop:
                # make sure we start with a big loss
                assert loss > 50
                first_loop = False
            logger.info(f"{loss=}, {TRICYCLE_CONTEXT.loss_scale_factor=}")
            model.update(optimiser)

        # make sure the loss has decreased as expected
        assert 7.5 < loss < 8
