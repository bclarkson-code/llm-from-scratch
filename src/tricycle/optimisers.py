"""
Optimisers for gradient-based optimisation.

This module contains various optimiser classes that can be used for
gradient-based optimisation of tensors.
"""

from logging import getLogger
from warnings import warn

from tricycle.context import TRICYCLE_CONTEXT
from tricycle.tensor import Tensor

LOGGER = getLogger(__name__)


class Optimiser:
    """Base class for optimisers."""

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Apply optimisation to the given tensor.

        Args:
            tensor (Tensor): The tensor to optimise.

        Returns:
            Tensor: The optimised tensor.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def _reset_grad(self, tensor: Tensor):
        """
        Reset the gradient information of the tensor.

        Args:
            tensor (Tensor): The tensor to reset.

        Returns:
            Tensor: The tensor with reset gradient information.
        """
        tensor.grad = None
        tensor.args = None
        tensor.back_fns = None
        return tensor


class StochasticGradientDescent(Optimiser):
    """
    Stochastic Gradient Descent (SGD) optimiser.

    This optimiser implements SGD with optional weight decay and momentum.

    Attributes:
        learning_rate (float): The learning rate for the optimiser.
        weight_decay (float | None): The weight decay factor.
        momentum (float | None): The momentum factor.
        logger: The logger instance.
        momentum_store (dict): Store for momentum values.
    """

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float | None = None,
        momentum: float | None = None,
        logger=LOGGER,
    ):
        """
        Initialise the SGD optimiser.

        Args:
            learning_rate (float): The learning rate for the optimiser.
            weight_decay (float | None, optional): The weight decay factor. Defaults to None.
            momentum (float | None, optional): The momentum factor. Defaults to None.
            logger (optional): The logger instance. Defaults to LOGGER.
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.logger = logger

        self.momentum_store = {}

    def update_weight(self, tensor: Tensor):
        """
        Perform a gradient update on a tensor.

        This method updates the tensor's weights using the calculated gradients,
        optionally applying weight decay and momentum.

        Args:
            tensor (Tensor): The tensor to update.

        Returns:
            Tensor: The updated tensor.
        """
        xp = tensor.xp
        assert tensor.grad is not None

        # We need to do gradient updates in full precision otherwise we get
        # stability issues
        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.grad.array = (
                tensor.grad.array.astype(xp.float32)
                / TRICYCLE_CONTEXT.loss_scale_factor
            )
            if not tensor.array.dtype == xp.float32:
                tensor.array = tensor.array.astype(xp.float32)

        if tensor.grad.is_batched:
            tensor.grad = tensor.grad.from_batched().einsum("z...->...")

        grad = self.learning_rate * tensor.grad.array

        if self.weight_decay is not None:
            wd = self.learning_rate * self.weight_decay * tensor.array
            grad += wd

        if self.momentum is not None and self.momentum > 0:
            if tensor._id not in self.momentum_store:
                last_momentum = tensor.xp.zeros(grad.shape)
            else:
                last_momentum = self.momentum_store[tensor._id]

            grad += self.momentum * last_momentum
            self.momentum_store[tensor._id] = grad

        # make sure our gradients aren't underflowing or overflow
        if not xp.isfinite(grad).all():
            warn(
                "Found nans in gradient, skipping this gradient and"
                "decreasing loss scaling. If this warning persists, "
                "check that your learning rate isn't too high"
            )
            TRICYCLE_CONTEXT.loss_scale_factor /= 2
            self.logger.warn(
                f"New scaling factor: {TRICYCLE_CONTEXT.loss_scale_factor}"
            )
            return tensor

        if (grad == 0).sum() > grad.size * 0.05:
            warn(
                "Found too many 0's in gradient, skipping this gradient and"
                "increasing loss scaling. If this warning persists, "
                "check that your learning rate isn't too low"
            )
            TRICYCLE_CONTEXT.loss_scale_factor *= 2
            self.logger.warn(
                f"New scaling factor: {TRICYCLE_CONTEXT.loss_scale_factor}"
            )
            return tensor

        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array -= grad.astype(xp.float32)

        tensor.grad.array.fill(0)
        return tensor

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Apply the SGD optimisation to the given tensor.

        Args:
            tensor (Tensor): The tensor to optimise.

        Returns:
            Tensor: The optimised tensor.
        """
        return self._reset_grad(self.update_weight(tensor))


class AdamW(Optimiser):
    """
    AdamW optimiser.

    This optimiser implements the AdamW algorithm, which is Adam with weight decay.

    Attributes:
        learning_rate (float): The learning rate for the optimiser.
        betas (tuple): The exponential decay rates for the moment estimates.
        eps (float): A small constant for numerical stability.
        weight_decay (float): The weight decay factor.
        timestep (int): The current time step.
        momentum (dict): Store for first moment estimates.
        square_momentum (dict): Store for second moment estimates.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
    ):
        """
        Initialise the AdamW optimiser.

        Args:
            learning_rate (float, optional): The learning rate. Defaults to 1e-3.
            betas (tuple, optional): The exponential decay rates for the moment estimates. Defaults to (0.9, 0.999).
            eps (float, optional): A small constant for numerical stability. Defaults to 1e-6.
            weight_decay (float, optional): The weight decay factor. Defaults to 0.01.
        """
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.timestep = 1

        self.momentum = {}
        self.square_momentum = {}

    def step(self):
        """
        Increase the time step.

        This method should be called after each optimisation step.
        """
        # we compute the updates dynamically so we'll need to remember to
        # call this
        self.timestep += 1

    def update_weight(self, tensor: Tensor) -> Tensor:
        """
        Perform a weight update on a tensor using the AdamW algorithm.

        Args:
            tensor (Tensor): The tensor to update.

        Returns:
            Tensor: The updated tensor.
        """
        key = tensor._id
        xp = tensor.xp

        assert tensor.grad is not None
        grad = tensor.grad.array

        if TRICYCLE_CONTEXT.use_mixed_precision:
            grad = grad.astype(xp.float32) / TRICYCLE_CONTEXT.loss_scale_factor
            if not tensor.array.dtype == xp.float32:
                tensor.array = tensor.array.astype(xp.float32)

        # initialise stores
        if key not in self.momentum:
            self.momentum[key] = xp.zeros_like(grad, dtype=xp.float32)
        if key not in self.square_momentum:
            self.square_momentum[key] = tensor.xp.zeros_like(
                grad, dtype=xp.float32
            )

        self.momentum[key] = (
            self.betas[0] * self.momentum[key] + (1 - self.betas[0]) * grad
        )

        self.square_momentum[key] = self.betas[1] * self.square_momentum[
            key
        ] + (1 - self.betas[1]) * (grad * grad)

        momentum_estimate = self.momentum[key] / (
            1 - self.betas[0] ** self.timestep
        )
        square_momentum_estimate = self.square_momentum[key] / (
            1 - self.betas[1] ** self.timestep
        )

        combined_grad = self.learning_rate * (
            momentum_estimate / (xp.sqrt(square_momentum_estimate) + self.eps)
            + self.weight_decay * tensor.array
        )

        # make sure our gradients aren't underflowing or overflow
        if not xp.isfinite(combined_grad).all():
            warn(
                "Found nans in gradient, skipping this gradient and"
                "decreasing loss scaling. If this warning persists, "
                "check that your learning rate isn't too high"
            )
            TRICYCLE_CONTEXT.loss_scale_factor /= 2
            self.logger.warn(
                f"New scaling factor: {TRICYCLE_CONTEXT.loss_scale_factor}"
            )
            return tensor

        if (combined_grad == 0).sum() > combined_grad.size * 0.05:
            warn(
                "Found too many 0's in gradient, skipping this gradient and"
                "increasing loss scaling. If this warning persists, "
                "check that your learning rate isn't too low"
            )
            TRICYCLE_CONTEXT.loss_scale_factor *= 2
            self.logger.warn(
                f"New scaling factor: {TRICYCLE_CONTEXT.loss_scale_factor}"
            )
            return tensor

        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array -= combined_grad.astype(xp.float32)

        tensor.grad.array.fill(0)
        return tensor

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Apply the AdamW optimisation to the given tensor.

        Args:
            tensor (Tensor): The tensor to optimise.

        Returns:
            Tensor: The optimised tensor.
        """
        return self._reset_grad(self.update_weight(tensor))
