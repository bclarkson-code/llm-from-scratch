"""Loss functions for neural network training.

This module contains implementations of common loss functions used in neural
network training, such as Mean Squared Error and Cross Entropy.

Classes:
    MeanSquaredError: Calculates the Mean Squared Error loss.
    CrossEntropy: Calculates the Cross Entropy loss.
"""

import logging

from tricycle.context import TRICYCLE_CONTEXT
from tricycle.ops import Op
from tricycle.tensor import Tensor

logger = logging.getLogger(__name__)


class MeanSquaredError(Op):
    """Calculates Mean Squared Error loss.

    This class implements the Mean Squared Error (MSE) loss function, which
    measures the average squared difference between the predicted and true values.

    Attributes:
        diff: The difference between predicted and true values.
        divisor: A scaling factor for the loss calculation.

    """

    def backward(self, grad: Tensor) -> Tensor:
        """Computes the backward pass for Mean Squared Error loss.

        Args:
            grad: A Tensor containing the gradient from the previous layer.

        Returns:
            A Tensor containing the computed gradients.
        """
        xp = grad.xp

        if TRICYCLE_CONTEXT.use_mixed_precision:
            grad.array = grad.array.astype(xp.float32)

        out = self.diff * 2 * grad.array * self.divisor

        if TRICYCLE_CONTEXT.use_mixed_precision:
            out = out.astype(xp.float16)

        return Tensor(out)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Computes the forward pass for Mean Squared Error loss.

        Args:
            y_true: A Tensor containing the true values.
            y_pred: A Tensor containing the predicted values.

        Returns:
            A Tensor containing the computed MSE loss.

        Raises:
            ValueError: If the computed loss is infinite.
        """
        xp = y_pred.xp

        if TRICYCLE_CONTEXT.use_mixed_precision:
            y_pred.array = y_pred.array.astype(xp.float32)
            y_true.array = y_true.array.astype(xp.float32)

        self.diff = y_pred.array - y_true.array
        self.divisor = 1 / xp.prod(y_pred.shape[-1])

        out = (self.diff**2).sum() * self.divisor

        if TRICYCLE_CONTEXT.use_mixed_precision:
            out *= TRICYCLE_CONTEXT.loss_scale_factor
            out = out.astype(xp.float16)

        if not xp.isfinite(out):
            raise ValueError("Loss is infinite")

        # only y_pred is differentiable: y_true is a constant
        return Tensor(
            out,
            args=(y_pred,),
            back_fns=(self.backward,),
            name="mean_squared_error",
        )


class CrossEntropy(Op):
    """Calculates Cross Entropy loss.

    This class implements the Cross Entropy loss function, which is commonly
    used for classification tasks. It computes the loss given logits and target
    indices (as opposed to one-hot encoded tensors).

    Attributes:
        _y_true: The true labels (cached for backward pass).
        _log_softmax_pred: The log softmax of predictions (cached for backward pass).
        _out: The computed loss (cached for backward pass).
        _grad: The computed gradients (cached for backward pass).
    """

    def log_softmax(self, tensor: Tensor):
        """Computes the log softmax of the input tensor.

        Args:
            tensor: A Tensor containing the input values.

        Returns:
            The log softmax of the input tensor.
        """
        xp = tensor.xp
        x_max = xp.max(tensor.array, axis=-1, keepdims=True)
        log_sum_exp = x_max + xp.log(
            xp.sum(xp.exp(tensor.array - x_max), axis=-1, keepdims=True)
        )
        return tensor.array - log_sum_exp

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Computes the forward pass for Cross Entropy loss.

        Args:
            y_true: A Tensor containing the true labels.
            y_pred: A Tensor containing the predicted logits.

        Returns:
            A Tensor containing the computed Cross Entropy loss.

        Raises:
            NotImplementedError: If the input tensor has an unsupported number of dimensions.
        """
        xp = y_pred.xp
        # cross entropy reduces a huge matrix to a single number which makes
        # it really sensitive to errors. To rememdy this, we need to use
        # full precision
        if TRICYCLE_CONTEXT.use_mixed_precision:
            y_pred.array = y_pred.array.astype(xp.float32)

        log_softmax_pred = self.log_softmax(y_pred)

        # Cache for backward pass
        self._y_true = y_true.array
        self._log_softmax_pred = log_softmax_pred

        ndim = log_softmax_pred.ndim

        if ndim == 3:
            batch_indices = xp.arange(y_true.shape[0], dtype=int)
            token_indices = xp.arange(y_true.shape[1], dtype=int)
            loss = -log_softmax_pred[
                batch_indices[:, None], token_indices, y_true.array
            ]
        elif ndim == 2:
            indices = xp.arange(y_true.shape[0], dtype=int)
            loss = -log_softmax_pred[indices, y_true.array]
        elif ndim == 1:
            loss = -log_softmax_pred[y_true.array]
        else:
            raise NotImplementedError(
                f"BinaryCrossEntropy with predictions with ndim: {ndim} are not yet supported"
            )

        # Mean loss over all elements
        loss = loss.mean()

        self._out = loss
        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._out = (
                self._out.astype(xp.float16)
                * TRICYCLE_CONTEXT.loss_scale_factor
            )

        return Tensor(
            self._out,
            is_batched=False,
            back_fns=(self.backward,),
            args=(y_pred,),
            name="cross_entropy",
        )

    def backward(self, grad: Tensor) -> Tensor:
        """Computes the backward pass for Cross Entropy loss.

        Args:
            grad: A Tensor containing the gradient from the previous layer.

        Returns:
            A Tensor containing the computed gradients.

        Raises:
            NotImplementedError: If the input tensor has an unsupported number of dimensions.
        """
        xp = grad.xp
        ndim = self._log_softmax_pred.ndim

        if TRICYCLE_CONTEXT.use_mixed_precision:
            grad.array = grad.array.astype(xp.float32)

        if ndim == 3:
            batch_indices = xp.arange(self._y_true.shape[0], dtype=int)
            token_indices = xp.arange(self._y_true.shape[1], dtype=int)
            grad_output = xp.exp(self._log_softmax_pred)
            grad_output[
                batch_indices[:, None], token_indices, self._y_true
            ] -= 1
            grad_output *= grad.array / (
                self._y_true.shape[0] * self._y_true.shape[1]
            )

        elif ndim == 2:
            indices = xp.arange(self._y_true.shape[0], dtype=int)
            grad_output = xp.exp(self._log_softmax_pred)
            grad_output[indices, self._y_true] -= 1
            grad_output *= grad.array / self._y_true.shape[0]
        elif ndim == 1:
            grad_output = xp.exp(self._log_softmax_pred)
            grad_output[self._y_true] -= 1
            grad_output *= grad.array
        else:
            raise NotImplementedError(
                f"BinaryCrossEntropy with predictions with ndim: {ndim} are not yet supported"
            )

        self._grad = grad_output

        # remember to convert the gradient back to the right precision
        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._grad = self._grad.astype(xp.float16)

        return Tensor(self._grad, is_batched=grad.is_batched)
