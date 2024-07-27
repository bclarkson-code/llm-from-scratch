"""Functions for neural network activation operations.

This module provides implementations of common activation functions
used in neural networks, including Softmax and Sigmoid.
"""

from tricycle.context import TRICYCLE_CONTEXT
from tricycle.ops import Op
from tricycle.tensor import Tensor


class Softmax(Op):
    """Applies the softmax function to the input tensor.

    The softmax function is applied only to the final dimension of the tensor.
    The input is normalized for numeric stability.

    Attributes:
        _out: The output of the forward pass.
        _grad: The gradient computed during the backward pass.
    """

    def back_fn(self, grad: Tensor) -> Tensor:
        """Computes the gradient of the softmax function.

        Args:
            grad: The gradient tensor from the subsequent layer.

        Returns:
            A Tensor containing the computed gradient.
        """
        xp = grad.xp

        inner = xp.sum(grad.array * self._out, axis=-1, keepdims=True)
        self._grad = self._out * (grad.array - inner)

        return Tensor(
            self._grad,
            is_batched=grad.is_batched,
            requires_grad=grad.requires_grad,
        )

    def forward(self, tensor: Tensor):
        """Applies the softmax function to the input tensor.

        Args:
            tensor: The input tensor.

        Returns:
            A Tensor with the softmax function applied.
        """
        xp = tensor.xp

        # Exponents tend to overflow/underflow when using 16 bit precision
        # so we need to switch to 32 bit
        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array = tensor.array.astype(xp.float32)

        exp = xp.exp(
            # subtract the largest value for numeric stability
            tensor.array
            - xp.max(tensor.array, axis=-1, keepdims=True)
        )
        denominator = xp.sum(exp, axis=-1, keepdims=True)
        self._out = exp / denominator
        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._out = self._out.astype(xp.float16)

        return Tensor(
            self._out,
            args=(tensor,),
            name="softmax",
            is_batched=tensor.is_batched,
            back_fns=(self.back_fn,),
        )


class Sigmoid(Op):
    """Applies the sigmoid function to the input tensor.

    Attributes:
        _out: The output of the forward pass.
        _grad: The gradient computed during the backward pass.
    """

    def backward(self, grad: Tensor) -> Tensor:
        """Computes the gradient of the sigmoid function.

        Args:
            grad: The gradient tensor from the subsequent layer.

        Returns:
            A Tensor containing the computed gradient.
        """
        self._grad = self._out * (1 - self._out) * grad.array
        return Tensor(self._grad, requires_grad=grad)

    def forward(self, tensor: Tensor) -> Tensor:
        """Applies the sigmoid function to the input tensor.

        Args:
            tensor: The input tensor.

        Returns:
            A Tensor with the sigmoid function applied.
        """
        xp = tensor.xp

        # Exponents tend to overflow/underflow when using 16 bit precision
        # so we need to switch to 32 bit
        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array = tensor.array.astype(xp.float32)

        self._out = 1 / (1 + xp.exp(-tensor.array))
        return Tensor(
            self._out,
            back_fns=(self.backward,),
            args=(tensor,),
            requires_grad=tensor.requires_grad,
        )
