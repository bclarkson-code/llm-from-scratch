"""
The Ops that have a single input and output are called Unary Ops.
Usually this means that an Operation is applied elementwise.

This file contains all of the unary operations in Tricycle.
"""

import numbers
from typing import Sequence

from numpy.typing import ArrayLike

from tricycle.context import TRICYCLE_CONTEXT
from tricycle.ops import Op
from tricycle.tensor import Tensor, select_backend
from tricycle.utils import shapes_match


def nothing(tensor):
    """
    Do nothing to a tensor.

    Operations like addition have a back_fn of multiplying by 1 which is
    equivalent to doing nothing. Instead of adding a bunch of checks to see
    whether a back_fn exists or not, the logic is much simpler by adding this
    function that does nothing.

    Args:
        tensor: The input tensor.

    Returns:
        The input tensor unchanged.
    """
    return tensor


class UnaryAdd(Op):
    """A class for adding a constant to a tensor elementwise."""

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Add a constant, elementwise, to a tensor. The constant is not
        differentiable.

        Args:
            tensor: The input tensor.
            constant: The constant to add.

        Returns:
            A new Tensor with the constant added elementwise.

        Raises:
            AssertionError: If tensor is not a Tensor or constant is not a number.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert isinstance(constant, numbers.Number)

        self._out = xp.add(tensor.array, constant)

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(nothing,),
            name=f"+ {constant}",
            is_batched=tensor.is_batched,
        )


class UnaryMultiply(Op):
    """A class for multiplying a tensor by a constant elementwise."""

    _constant: float

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the multiplication operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        xp = grad.xp

        self._grad = xp.multiply(grad.array, self._constant)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Multiply a constant, elementwise, to a tensor. The constant is not
        differentiable.

        Args:
            tensor: The input tensor.
            constant: The constant to multiply.

        Returns:
            A new Tensor with the constant multiplied elementwise.

        Raises:
            AssertionError: If tensor is not a Tensor or constant is not a scalar.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.multiply(tensor.array, constant)
        self._constant = constant

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name=f"+ {constant}",
            is_batched=tensor.is_batched,
        )


class UnarySubtract(Op):
    """A class for subtracting a constant from a tensor elementwise."""

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Subtract a constant, elementwise, from a tensor. The constant is not
        differentiable.

        Args:
            tensor: The input tensor.
            constant: The constant to subtract.

        Returns:
            A new Tensor with the constant subtracted elementwise.
        """
        return UnaryAdd()(tensor, -constant)


class UnaryPower(Op):
    """A class for raising a tensor to a constant power elementwise."""

    _input: ArrayLike
    _constant: float

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the power operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        xp = grad.xp

        self._grad = xp.power(self._input, self._constant - 1)
        self._grad *= self._constant * grad.array

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Raise a tensor to a constant, elementwise. The constant is not
        differentiable.

        Args:
            tensor: The input tensor.
            constant: The constant power.

        Returns:
            A new Tensor with each element raised to the constant power.

        Raises:
            AssertionError: If tensor is not a Tensor or constant is not a scalar.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.power(tensor.array, constant)
        self._input = tensor.array
        self._constant = constant

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name=f"^ {constant}",
            is_batched=tensor.is_batched,
        )


class UnaryDivide(Op):
    """A class for dividing a constant by a tensor elementwise."""

    # TODO: manually define the derivative instead of using other operations
    def forward(self, constant: float | int, tensor: Tensor) -> Tensor:
        """
        Divide a constant by a tensor, elementwise. The constant is not
        differentiable.

        Args:
            constant: The constant numerator.
            tensor: The tensor denominator.

        Returns:
            A new Tensor with the constant divided by each element of the tensor.
        """
        upow = UnaryPower()
        umul = UnaryMultiply()

        xp = tensor.xp
        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array = tensor.array.astype(xp.float32)

        out = umul(upow(tensor, -1.0), constant)

        if TRICYCLE_CONTEXT.use_mixed_precision:
            out.array = out.array.astype(xp.float16)
        return out


class UnaryMax(Op):
    """A class for computing the elementwise maximum of a tensor and a constant."""

    _is_bigger: ArrayLike

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the maximum operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        self._grad = grad.array * self._is_bigger

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Return the max of the tensor and the constant,
        elementwise. The constant is not differentiable.

        Args:
            tensor: The input tensor.
            constant: The constant to compare against.

        Returns:
            A new Tensor with the elementwise maximum of the input and the constant.

        Raises:
            AssertionError: If tensor is not a Tensor or constant is not a scalar.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.maximum(tensor.array, constant, dtype=tensor.dtype)

        self._is_bigger = tensor.array > constant

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name=f"> {constant}",
            is_batched=tensor.is_batched,
        )


class UnaryMin(Op):
    """A class for computing the elementwise minimum of a tensor and a constant."""

    _is_smaller: Tensor

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the minimum operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        self._grad = grad.array * self._is_smaller

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Return the min of the tensor and the constant,
        elementwise. The constant is not differentiable.

        Args:
            tensor: The input tensor.
            constant: The constant to compare against.

        Returns:
            A new Tensor with the elementwise minimum of the input and the constant.

        Raises:
            AssertionError: If tensor is not a Tensor or constant is not a scalar.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.minimum(tensor.array, constant, dtype=tensor.dtype)

        self._is_smaller = tensor.array < constant

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name=f"< {constant}",
            is_batched=tensor.is_batched,
        )


class UnaryExp(Op):
    """A class for computing the exponential of a tensor elementwise."""

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the exponential operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        self._grad = grad.array * self._out

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Raise every element of a tensor to the power of e.

        Args:
            tensor: The input tensor.

        Returns:
            A new Tensor with the exponential of each element.
        """
        xp = tensor.xp

        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array = tensor.array.astype(xp.float32)
        self._out = xp.exp(tensor.array)

        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._out = self._out.astype(xp.float16)

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="exp",
            is_batched=tensor.is_batched,
        )


class UnaryLog(Op):
    """A class for computing the natural logarithm of a tensor elementwise."""

    REALLY_SMALL_NUMBER = 1e-6

    _input: ArrayLike

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the logarithm operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        xp = grad.xp

        # add a small number to avoid dividing by 0
        denominator = self._input + self.REALLY_SMALL_NUMBER
        self._grad = grad.array * xp.divide(1, denominator)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Compute the natural logarithm of each element in the tensor.

        Args:
            tensor: The input tensor.

        Returns:
            A new Tensor with the natural logarithm of each element.
        """
        xp = tensor.xp

        self._out = xp.log(tensor.array)
        self._input = tensor.array

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="log",
            is_batched=tensor.is_batched,
        )


class UnarySin(Op):
    """A class for computing the sine of a tensor elementwise."""

    _input: ArrayLike

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the sine operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        xp = grad.xp

        self._grad = grad.array * xp.cos(self._input)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Applies the sine function, elementwise, to a tensor.

        Args:
            tensor: The input tensor.

        Returns:
            A new Tensor with the sine of each element.
        """
        xp = tensor.xp

        self._out = xp.sin(tensor.array)
        self._input = tensor.array

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="sin",
            is_batched=tensor.is_batched,
        )


class UnaryCos(Op):
    """A class for computing the cosine of a tensor elementwise."""

    _input: ArrayLike

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the cosine operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        xp = grad.xp

        self._grad = grad.array * -xp.sin(self._input)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Applies the cosine function, elementwise, to a tensor.

        Args:
            tensor: The input tensor.

        Returns:
            A new Tensor with the cosine of each element.
        """
        xp = tensor.xp

        self._out = xp.cos(tensor.array)
        self._input = tensor.array

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="cos",
            is_batched=tensor.is_batched,
        )


class UnarySquareRoot(Op):
    """A class for computing the square root of a tensor elementwise."""

    # TODO: This would probably be faster if we use xp.sqrt instead of xp.power
    def forward(self, tensor: Tensor):
        """
        Apply the square root function.

        Args:
            tensor: The input tensor.

        Returns:
            A new Tensor with the square root of each element.
        """
        upow = UnaryPower()
        return upow(tensor, 0.5)


class UnarySum(Op):
    """A class for summing all values in a tensor."""

    _in_shape: Sequence[int]
    _is_batched: bool

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the sum operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.

        Raises:
            AssertionError: If grad is not a scalar or an empty tensor.
        """
        xp = grad.xp

        assert xp.isscalar(grad) or grad.shape == ()

        self._grad = xp.full(self._in_shape, grad.array)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Sums all the values in a tensor.
        Note, this function always produces a non-batched output.

        Args:
            tensor: The input tensor.

        Returns:
            A new Tensor with the sum of all elements.

        Raises:
            AssertionError: If the sum operation fails.
        """
        xp = tensor.xp

        self._out = xp.sum(tensor.array)
        self._in_shape = tensor.shape
        self._is_batched = tensor.is_batched

        assert self._out is not None

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="sum",
            is_batched=False,
        )


class UnaryMask(Op):
    """A class for applying a binary mask to a tensor."""

    _mask: ArrayLike

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the mask operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        xp = select_backend(grad.array, self._mask)

        self._grad = xp.where(self._mask, grad.array, 0)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, mask: Tensor) -> Tensor:
        """
        Apply a binary mask to a numpy array, setting values to 0 where
        the mask is True.

        Args:
            tensor: The input tensor.
            mask: The binary mask tensor.

        Returns:
            A new Tensor with the mask applied.

        Raises:
            AssertionError: If tensor and mask shapes don't match or if mask requires grad.
        """
        xp = select_backend(tensor.array, mask.array)
        assert shapes_match(tensor, mask)
        assert (
            not mask.requires_grad
        ), "Cannot compute gradient of a binary mask"

        self._out = xp.where(mask.array, tensor.array, 0)
        self._mask = mask.array

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="umask",
            is_batched=tensor.is_batched,
        )


def batch(tensor: Tensor) -> Tensor:
    """
    Tell Tricycle to treat this tensor as a batch of tensors.

    Args:
        tensor: The input tensor.

    Returns:
        A new Tensor marked as batched.
    """
    if tensor.is_batched:
        return tensor

    return Tensor(
        array=tensor.array,
        name="batch",
        requires_grad=tensor.requires_grad,
        is_batched=True,
        args=(tensor,),
        back_fns=(unbatch,),
    )


def unbatch(tensor: Tensor) -> Tensor:
    """
    Tell Tricycle to treat this tensor as a single tensor
    (not a batch of tensors).

    Args:
        tensor: The input tensor.

    Returns:
        A new Tensor marked as unbatched.
    """
    if not tensor.is_batched:
        return tensor

    return Tensor(
        array=tensor.array,
        name="unbatch",
        requires_grad=tensor.requires_grad,
        is_batched=False,
        args=(tensor,),
        back_fns=(batch,),
    )


class Batch(Op):
    """A class for batching a tensor."""

    is_batched: bool

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the batch operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        return Tensor(
            array=grad.array,
            name="unbatch",
            requires_grad=grad.requires_grad,
            is_batched=self.is_batched,
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Mark a tensor as batched.

        Args:
            tensor: The input tensor.

        Returns:
            A new Tensor marked as batched.
        """
        self.is_batched = tensor.is_batched

        return Tensor(
            array=tensor.array,
            name="batch",
            requires_grad=tensor.requires_grad,
            is_batched=True,
            args=(tensor,),
            back_fns=(self.back_fn,),
            dtype=tensor.dtype,
        )


class Unbatch(Op):
    """A class for unbatching a tensor."""

    is_batched: bool

    def back_fn(self, grad: Tensor) -> Tensor:
        """
        Compute the gradient for the unbatch operation.

        Args:
            grad: The gradient tensor.

        Returns:
            The computed gradient.
        """
        return Tensor(
            array=grad.array,
            name="batch",
            requires_grad=grad.requires_grad,
            is_batched=self.is_batched,
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Mark a tensor as unbatched.

        Args:
            tensor: The input tensor.

        Returns:
            A new Tensor marked as unbatched.
        """
        self.is_batched = tensor.is_batched

        return Tensor(
            array=tensor.array,
            name="unbatch",
            requires_grad=tensor.requires_grad,
            is_batched=False,
            args=(tensor,),
            back_fns=(self.back_fn,),
            dtype=tensor.dtype,
        )
