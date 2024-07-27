"""
Binary operations for the Tricycle framework.

This module contains binary operations that can be applied to tensors of the same shape.
These operations include element-wise addition, subtraction, multiplication, division,
and comparison operations like maximum and minimum.

Note:
    In Tricycle, binary operations are only allowed on matrices of the same shape to
    simplify gradient computations.
"""

from numpy.typing import ArrayLike

from tricycle.ops import Einsum, Op
from tricycle.tensor import Tensor, select_backend
from tricycle.unary import UnaryDivide, nothing
from tricycle.utils import shapes_match


class BinaryAdd(Op):
    """Element-wise addition of two tensors.

    This class implements the forward pass for element-wise addition of two tensors.

    Attributes:
        _out: The output of the forward pass.
    """

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """Add two tensors element-wise.

        Args:
            tensor_1: First input tensor.
            tensor_2: Second input tensor.

        Returns:
            A Tensor representing the element-wise sum of the input tensors.

        Raises:
            AssertionError: If the shapes of the input tensors do not match.
        """
        xp = select_backend(tensor_1.array, tensor_2.array)

        assert shapes_match(tensor_1, tensor_2)
        self._out = xp.add(tensor_1.array, tensor_2.array)

        return Tensor(
            self._out,
            args=(tensor_1, tensor_2),
            back_fns=(nothing, nothing),
            name="badd",
            is_batched=tensor_1.is_batched or tensor_2.is_batched,
        )


class BinarySubtract(Op):
    """Element-wise subtraction of two tensors.

    This class implements the forward and backward passes for element-wise subtraction
    of two tensors.

    Attributes:
        _grad: The gradient of the backward pass for the second tensor.
        _out: The output of the forward pass.
    """

    def back_fn_2(self, grad: Tensor) -> Tensor:
        """Compute the gradient for the second tensor in the subtraction.

        Args:
            grad: The gradient tensor.

        Returns:
            A Tensor representing the gradient for the second tensor.
        """
        self._grad = -grad.array

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """Subtract one tensor from another element-wise.

        The two tensors must have the same shape.

        Args:
            tensor_1: First input tensor.
            tensor_2: Second input tensor to be subtracted from the first.

        Returns:
            A Tensor representing the element-wise difference of the input tensors.

        Raises:
            AssertionError: If the shapes of the input tensors do not match.
        """
        xp = select_backend(tensor_1.array, tensor_2.array)

        assert shapes_match(tensor_1, tensor_2)
        self._out = xp.subtract(tensor_1.array, tensor_2.array)

        return Tensor(
            self._out,
            args=(tensor_1, tensor_2),
            back_fns=(nothing, self.back_fn_2),
            name="bsub",
            is_batched=tensor_1.is_batched or tensor_2.is_batched,
        )


class BinaryMultiply(Op):
    """Element-wise multiplication of two tensors.

    This class implements the forward pass for element-wise multiplication of two tensors.
    """

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """Multiply the elements of two tensors together, element-wise.

        The two tensors must have the same shape.

        Args:
            tensor_1: First input tensor.
            tensor_2: Second input tensor.

        Returns:
            A Tensor representing the element-wise product of the input tensors.

        Raises:
            AssertionError: If the shapes of the input tensors do not match.
        """
        assert shapes_match(tensor_1, tensor_2)

        result = Einsum("...,...->...")(tensor_1, tensor_2)
        result.name = "bmul"
        return result


class BinaryDivide(Op):
    """Element-wise division of two tensors.

    This class implements the forward pass for element-wise division of two tensors.

    TODO: we should probably fuse these into a single op
    """

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """Divide the elements of two tensors together, element-wise.

        The two tensors must have the same shape.

        Args:
            tensor_1: First input tensor (numerator).
            tensor_2: Second input tensor (denominator).

        Returns:
            A Tensor representing the element-wise division of the input tensors.

        Raises:
            AssertionError: If the shapes of the input tensors do not match.
        """
        mul = BinaryMultiply()
        div = UnaryDivide()

        return mul(tensor_1, div(1, tensor_2))


class BinaryMax(Op):
    """Element-wise maximum of two tensors.

    This class implements the forward and backward passes for element-wise maximum
    of two tensors.

    Attributes:
        _is_bigger_1: Boolean array indicating where the first tensor is larger.
        _is_bigger_2: Boolean array indicating where the second tensor is larger or equal.
        _out: The output of the forward pass.
        _grad_1: The gradient for the first tensor.
        _grad_2: The gradient for the second tensor.
    """

    def back_fn_1(self, grad: Tensor) -> Tensor:
        """Compute the gradient for the first tensor in the maximum operation.

        Args:
            grad: The gradient tensor.

        Returns:
            A Tensor representing the gradient for the first tensor.
        """
        self._grad_1 = grad.array * self._is_bigger_1

        return Tensor(array=self._grad_1, is_batched=grad.is_batched)

    def back_fn_2(self, grad: Tensor) -> Tensor:
        """Compute the gradient for the second tensor in the maximum operation.

        Args:
            grad: The gradient tensor.

        Returns:
            A Tensor representing the gradient for the second tensor.
        """
        self._grad_2 = grad.array * self._is_bigger_2

        return Tensor(array=self._grad_2, is_batched=grad.is_batched)

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """Compare two tensors element-wise, returning the maximum of each pair of elements.

        The two tensors must have the same shape. If elements are equal, return the first.

        Args:
            tensor_1: First input tensor.
            tensor_2: Second input tensor.

        Returns:
            A Tensor representing the element-wise maximum of the input tensors.

        Raises:
            AssertionError: If the shapes of the input tensors do not match.
        """
        xp = select_backend(tensor_1.array, tensor_2.array)
        assert shapes_match(tensor_1, tensor_2)

        self._out = xp.maximum(tensor_1.array, tensor_2.array)

        self._is_bigger_1 = tensor_1.array > tensor_2.array
        self._is_bigger_2 = tensor_1.array <= tensor_2.array

        return Tensor(
            self._out,
            args=(tensor_1, tensor_2),
            back_fns=(self.back_fn_1, self.back_fn_2),
            name="bmax",
            is_batched=tensor_1.is_batched or tensor_2.is_batched,
        )


class BinaryMin(Op):
    """Element-wise minimum of two tensors.

    This class implements the forward and backward passes for element-wise minimum
    of two tensors.

    Attributes:
        _is_smaller_1: Boolean array indicating where the first tensor is smaller.
        _is_smaller_2: Boolean array indicating where the second tensor is smaller or equal.
        _out: The output of the forward pass.
        _grad_1: The gradient for the first tensor.
        _grad_2: The gradient for the second tensor.
    """

    def back_fn_1(self, grad: Tensor) -> Tensor:
        """Compute the gradient for the first tensor in the minimum operation.

        Args:
            grad: The gradient tensor.

        Returns:
            A Tensor representing the gradient for the first tensor.
        """
        self._grad_1 = grad.array * self._is_smaller_1

        return Tensor(array=self._grad_1, is_batched=grad.is_batched)

    def back_fn_2(self, grad: Tensor) -> Tensor:
        """Compute the gradient for the second tensor in the minimum operation.

        Args:
            grad: The gradient tensor.

        Returns:
            A Tensor representing the gradient for the second tensor.
        """
        self._grad_2 = grad.array * self._is_smaller_2

        return Tensor(array=self._grad_2, is_batched=grad.is_batched)

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """Compare two tensors element-wise, returning the minimum of each pair of elements.

        The two tensors must have the same shape. If elements are equal, return the first.

        Args:
            tensor_1: First input tensor.
            tensor_2: Second input tensor.

        Returns:
            A Tensor representing the element-wise minimum of the input tensors.

        Raises:
            AssertionError: If the shapes of the input tensors do not match.
        """
        xp = select_backend(tensor_1.array, tensor_2.array)
        assert shapes_match(tensor_1, tensor_2)

        self._out = xp.minimum(tensor_1.array, tensor_2.array)

        self._is_smaller_1 = tensor_1.array < tensor_2.array
        self._is_smaller_2 = tensor_1.array >= tensor_2.array

        return Tensor(
            self._out,
            args=(tensor_1, tensor_2),
            back_fns=(self.back_fn_1, self.back_fn_2),
            name="bmin",
            is_batched=tensor_1.is_batched or tensor_2.is_batched,
        )
