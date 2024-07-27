"""Operations module for tensor manipulations.

This module contains various operations that can be applied to tensors,
including repeat, split, reshape, and mean operations.
"""

from abc import abstractmethod
from typing import Sequence

from numpy.typing import ArrayLike

from tricycle.context import TRICYCLE_CONTEXT
from tricycle.einsum import Einsum, Subscript
from tricycle.tensor import Tensor


class Op:
    """Base class for operations."""

    _out: ArrayLike | None = None

    def __call__(self, *args, **kwargs) -> Tensor:
        """Call the forward method of the operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the forward operation.
        """
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Abstract method for the forward pass of the operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.

        Returns:
            Tensor: The result of the forward operation.
        """
        raise NotImplementedError()


class Repeat(Op):
    """Operation to repeat a tensor along its final axis."""

    def forward(self, tensor: Tensor, repeats: int):
        """Repeat a tensor along its final axis.

        This is done by multiplying with a ones tensor the same shape as the
        desired output.

        Args:
            tensor (Tensor): The input tensor to repeat.
            repeats (int): The number of times to repeat the tensor.

        Returns:
            Tensor: The repeated tensor.
        """
        xp = tensor.xp
        subscript = Subscript("...,...a->...a")
        new_shape = tensor.shape + (repeats,)
        ones = Tensor(
            xp.ones(new_shape),
            is_batched=tensor.is_batched,
            requires_grad=False,
        )

        return Einsum(subscript)(tensor, ones)


class Split(Op):
    """Operation to split a tensor along an axis."""

    _indices: tuple[int]
    _axis: int
    _n_splits: int
    _grad: list[ArrayLike]

    def back_fn(self, grad: Tensor, idx: int) -> Tensor:
        """The backwards operation for a split operation.

        Produces a tensor of zeros the same shape as the input
        except in the section that was split.

        Args:
            grad (Tensor): The gradient tensor.
            idx (int): The index of the split.

        Returns:
            Tensor: The gradient for the input tensor.

        Example:
            >>> result = split([1,2,3,4], 2)
            >>> result
            [tensor([1, 2]), tensor([3, 4])]
            # set an arbitrary derivative for first split
            >>> result[0].grad = Tensor([1,1])
            >>> undo_split(result[0].grad)
            [1, 1, 0, 0]
        """
        xp = grad.xp
        self._grad[idx] = xp.zeros(self._in_shape)

        # TODO: this loop is really slow and should be replaced
        indices = []
        for i in range(self._grad[idx].ndim):
            if i == self._axis % self._grad[idx].ndim:
                step = self._in_shape[i] // self._n_splits
                start = step * idx
                end = step * (idx + 1)
                indices.append(slice(start, end))
            else:
                indices.append(slice(None))
        self._grad[idx][tuple(indices)] = grad.array

        result = Tensor(self._grad[idx])
        result.is_batched = grad.is_batched
        return result

    def forward(
        self, tensor: Tensor, n_splits: int, axis: int = -1
    ) -> Sequence[Tensor]:
        """Split a tensor along an axis into n_splits partitions.

        Args:
            tensor (Tensor): The input tensor to split.
            n_splits (int): The number of splits to make.
            axis (int, optional): The axis along which to split. Defaults to -1.

        Returns:
            Sequence[Tensor]: A sequence of split tensors.
        """
        xp = tensor.xp

        assert isinstance(n_splits, int)

        self._out = xp.split(tensor.array, n_splits, axis=axis)
        self._in_shape = tensor.shape
        self._axis = axis
        self._n_splits = n_splits
        self._grad = [None] * n_splits

        # TODO: this loop is really slow and should be replaced
        results = []
        for idx, result in enumerate(self._out):
            # the back_fn depends on index so we need to
            # dynamically create this function
            def back_fn(grad, idx=idx):
                return self.back_fn(grad, idx=idx)

            result = Tensor(result)
            result.back_fns = (back_fn,)
            result.args = (tensor,)
            result.is_batched = tensor.is_batched
            results.append(result)
        return results


class Reshape(Op):
    """Operation to reshape a tensor."""

    _original_shape: Sequence[int]

    def back_fn(self, grad: Tensor) -> Tensor:
        """Backward function for the reshape operation.

        Args:
            grad (Tensor): The gradient tensor.

        Returns:
            Tensor: The gradient reshaped to the original shape.
        """
        xp = grad.xp

        self._grad = xp.reshape(grad.array, self._original_shape)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, shape: Sequence[int]) -> Tensor:
        """Reshape a tensor.

        The new shape needs to have the same number of elements
        as the original, but can have any number of dimensions.

        Args:
            tensor (Tensor): The input tensor to reshape.
            shape (Sequence[int]): The new shape for the tensor.

        Returns:
            Tensor: The reshaped tensor.
        """
        xp = tensor.xp

        # if the tensor is batched, don't include the first dimension in
        # the reshape
        if tensor.is_batched:
            shape = [tensor.shape[0]] + list(shape)

        self._out = xp.reshape(tensor.array, shape)
        self._original_shape = tensor.shape

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="reshape",
            is_batched=tensor.is_batched,
        )


class Mean(Op):
    """Operation to find the mean of a tensor."""

    def backward(self, grad: Tensor) -> Tensor:
        """Backward function for the mean operation.

        Args:
            grad (Tensor): The gradient tensor.

        Returns:
            Tensor: The gradient for the input tensor.
        """
        xp = grad.xp

        result = xp.full(self._in_shape, self.divisor)
        out = grad.array * result

        return Tensor(out, is_batched=self._is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """Find the mean of a tensor.

        Args:
            tensor (Tensor): The input tensor.

        Returns:
            Tensor: A tensor containing the mean value.
        """
        xp = tensor.xp
        self._is_batched = tensor.is_batched
        self._in_shape = tensor.shape

        # we can overflow here with large arrays so we'll use full precision
        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array = tensor.array.astype(xp.float32)

        self.divisor = 1 / xp.prod(tensor.shape) if tensor.shape else 1
        out = tensor.array.sum() * self.divisor

        if TRICYCLE_CONTEXT.use_mixed_precision:
            out = out.astype(xp.float16)

        return Tensor(
            out, name="mean", back_fns=(self.backward,), args=(tensor,)
        )
