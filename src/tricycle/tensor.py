"""
The core of Tricycle is the Tensor object, which is implemented in this file.

A Tensor is a wrapper around a numpy/cupy array that adds automatic
differentiation.

The autodiff algorithm itself can be found in `Tensor.backward`.

This file also contains a few other helpful functions like `batch` which
converts tensors to batched tensors.
"""

import logging
import numbers
import uuid
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from tricycle import GPU_ENABLED
from tricycle.context import TRICYCLE_CONTEXT
from tricycle.exceptions import GPUDisabledException
from tricycle.weakset import WeakSet

if TYPE_CHECKING:
    from tricycle.ops import Op

logger = logging.getLogger(__name__)

DEFAULT_DTYPE = np.float32


class Tensor:
    """
    An N-dimensional grid of numbers. This is implemented as a subclass
    of a standard numpy array.

    Attributes:
        _id (int): Unique identifier for the tensor.
        _parents (set[Tensor] | None): Parent tensors in the computation graph.
        array (ArrayLike): The underlying numpy/cupy array.
        args (tuple[Tensor, ...] | None): Arguments used to create this tensor.
        back_fns (tuple[Op, ...] | None): Backward functions for gradient computation.
        grad (Optional[Tensor]): Gradient of this tensor.
        name (Optional[str]): Name of the tensor.
        requires_grad (bool): Whether this tensor requires gradient computation.
        is_batched (bool): Whether this tensor is batched.
    """

    def __init__(
        self,
        array: ArrayLike,
        requires_grad: bool = True,
        is_batched: bool = False,
        args: tuple["Tensor", ...] | None = None,
        back_fns: tuple["Op", ...] | None = None,
        dtype: np.typing.DTypeLike = None,
        name: str | None = None,
        _id: int | None = None,
    ):
        """
        Initializes a new Tensor object.

        Args:
            array (ArrayLike): The underlying numpy/cupy array.
            requires_grad (bool, optional): Whether this tensor requires gradient computation. Defaults to True.
            is_batched (bool, optional): Whether this tensor is batched. Defaults to False.
            args (tuple[Tensor, ...] | None, optional): Arguments used to create this tensor. Defaults to None.
            back_fns (tuple[Op, ...] | None, optional): Backward functions for gradient computation. Defaults to None.
            dtype (np.typing.DTypeLike, optional): Data type of the tensor. Defaults to None.
            name (str | None, optional): Name of the tensor. Defaults to None.
            _id (int | None, optional): Unique identifier for the tensor. Defaults to None.
        """
        if isinstance(array, Tensor):
            self = array
            return
        self._id = _id or uuid.uuid4().int
        self._parents = None
        if GPU_ENABLED:
            import cupy

            if isinstance(array, (np.ndarray, cupy.ndarray)):
                self.array = array
            else:
                self.array = np.array(array)
        else:
            self.array = np.array(array)

        if dtype is None:
            if TRICYCLE_CONTEXT.use_mixed_precision:
                dtype = np.float16
            else:
                dtype = DEFAULT_DTYPE

        self.array = self.array.astype(dtype)
        self.grad = None

        self.requires_grad = requires_grad
        self.is_batched = is_batched
        self.args = args
        self.back_fns = back_fns
        self.name = name

    def _attach_parents(self):
        """
        Traverses through the graph, labelling each tensor with the tensors that
        are direct parents to it in the graph.

        This is done to enable traversal through the graph later in
        topological order.
        """
        stack: list["Tensor"] = [self]

        while stack:
            node = stack.pop()

            if not node.args:
                continue

            for arg in node.args:
                if not arg.requires_grad:
                    continue

                if arg._parents is None:
                    # if we use a set, we get a circular reference
                    # which can't be garbage collected, leading to a memory
                    # leak so we need to do a weakref to avoid the circular
                    # reference
                    arg._parents = WeakSet()

                # if a node has a parent we haven't visited yet, store it
                if node not in arg._parents:
                    stack.append(arg)
                    arg._parents.add(node)

    def _calculate_gradients(self, clip: float | None = None):
        """
        Calculates gradients for the computation graph.

        This method implements the backpropagation algorithm, traversing the graph
        from the output to the inputs and applying the chain rule to compute gradients.

        Args:
            clip (float | None, optional): Maximum absolute value for gradient clipping. Defaults to None.
        """
        self.grad = Tensor(
            self.xp.ones(self.array.shape, dtype=self.dtype),
            requires_grad=False,
            is_batched=self.is_batched,
        )

        stack: list["Tensor"] = [self]

        while stack:
            node = stack.pop()

            # if we have reached an input, we're done along this path
            if node.args is None or node.back_fns is None:
                continue

            for arg, back_fns in zip(node.args, node.back_fns):
                # if we reach a tensor that does not need gradient computation
                # (e.g a constant) then we're done along this path
                if not arg.requires_grad:
                    continue

                if arg._parents is None:
                    raise ValueError(
                        "arg.parents is None. Parents must be attached",
                        "before calculating gradients. Did you forget to ",
                        "call _attach_parents?",
                    )

                # already visited along this edge, dont do it again
                if node not in arg._parents:
                    continue

                arg._parents.remove(node)

                try:
                    # actuall calculate gradient for this node
                    grad = back_fns(node.grad)

                    # gradient clipping
                    # TODO: allow clipping by norm instead of just by value
                    if clip is not None:
                        grad.array = grad.xp.clip(grad.array, -clip, clip)

                    # add current gradient to any gradients we have already
                    # calculated for this node
                    if arg.grad is None:
                        arg.grad = grad
                    else:
                        arg.grad.array += grad.array

                except Exception as e:
                    raise e

                # only move to a new node if we have been to all of its parents
                if len(arg._parents) == 0:
                    # get rid of the weakref once we're done with a node so we
                    # can pickle the model. Weakrefs can't be pickled
                    arg._parents = None
                    stack.append(arg)

    def backward(self, clip: float | None = None):
        """
        Performs a backward pass through the graph, calculating the gradient
        for each parameter.

        Args:
            clip (float | None, optional): Maximum absolute value for gradient clipping. Defaults to None.
        """
        self._attach_parents()
        self._calculate_gradients(clip=clip)

    def __hash__(self) -> int:
        return self._id

    def __add__(self, other: Union[float, "Tensor"]) -> "Tensor":
        """
        Implements addition for Tensor objects.

        Args:
            other (Union[float, Tensor]): The value to add to this tensor.

        Returns:
            Tensor: The result of the addition.

        Raises:
            NotImplementedError: If addition is not supported between the given types.
        """
        if isinstance(other, numbers.Number):
            from tricycle.unary import UnaryAdd

            return UnaryAdd()(self, other)
        elif isinstance(other, Tensor):
            from tricycle.binary import BinaryAdd

            return BinaryAdd()(self, other)
        else:
            raise NotImplementedError(
                f"Cannot add {type(self)} and {type(other)}"
            )

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        """
        Implements subtraction for Tensor objects.

        Args:
            other (Union[float, Tensor]): The value to subtract from this tensor.

        Returns:
            Tensor: The result of the subtraction.

        Raises:
            NotImplementedError: If subtraction is not supported between the given types.
        """
        if isinstance(other, self.xp.ndarray) and not isinstance(
            other, Tensor
        ):
            other = Tensor(other)
        if self.xp.isscalar(other):
            from tricycle.unary import UnarySubtract

            return UnarySubtract()(self, other)
        elif isinstance(other, Tensor):
            from tricycle.binary import BinarySubtract

            return BinarySubtract()(self, other)

        else:
            raise NotImplementedError(
                f"Cannot sub {type(self)} and {type(other)}"
            )

    def __rsub__(self, other):
        return -(self - other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        """
        Implements multiplication for Tensor objects.

        Args:
            other (Union[float, Tensor]): The value to multiply with this tensor.

        Returns:
            Tensor: The result of the multiplication.

        Raises:
            NotImplementedError: If multiplication is not supported between the given types.
        """
        if isinstance(other, self.xp.ndarray) and not isinstance(
            other, Tensor
        ):
            other = Tensor(other)
        if self.xp.isscalar(other) or other.shape == ():
            from tricycle.unary import UnaryMultiply

            return UnaryMultiply()(self, other)

        elif isinstance(other, Tensor):
            from tricycle.binary import BinaryMultiply

            return BinaryMultiply()(self, other)

        else:
            raise NotImplementedError(
                f"Cannot mul {type(self)} and {type(other)}"
            )

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        """
        Implements true division for Tensor objects.

        Args:
            other (Union[float, Tensor]): The value to divide this tensor by.

        Returns:
            Tensor: The result of the division.

        Raises:
            NotImplementedError: If division is not supported between the given types.
        """
        if self.xp.isscalar(other):
            from tricycle.unary import UnaryMultiply

            return UnaryMultiply()(self, 1 / other)
        elif isinstance(other, Tensor):
            from tricycle.binary import BinaryDivide

            return BinaryDivide()(self, other)

        else:
            raise NotImplementedError(
                f"Cannot divide {type(self)} and {type(other)}"
            )

    def __rtruediv__(self, other):
        if self.xp.isscalar(other):
            from tricycle.unary import UnaryDivide

            return UnaryDivide()(other, self)
        elif isinstance(other, Tensor):
            from tricycle.binary import BinaryDivide

            return BinaryDivide()(other, self)

    def __itruediv__(self, other):
        return self / other

    def __pow__(self, other) -> "Tensor":
        """
        Implements exponentiation for Tensor objects.

        Args:
            other (Union[float, Tensor]): The exponent.

        Returns:
            Tensor: The result of the exponentiation.

        Raises:
            NotImplementedError: If exponentiation is not supported between the given types.
        """
        if isinstance(other, self.xp.ndarray) and not isinstance(
            other, Tensor
        ):
            other = Tensor(other)
        if self.xp.isscalar(other):
            from tricycle.unary import UnaryPower

            return UnaryPower()(self, other)
        elif isinstance(other, Tensor):
            raise NotImplementedError(
                "Cannot power two tensors of shape: "
                f"{self.shape}, {other.shape}"
            )
        else:
            raise NotImplementedError(
                f"Cannot power {type(self)} and {type(other)}"
            )

    def __lt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array < other.array)

        return Tensor(self.array < other)

    def __le__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array <= other.array)

        return Tensor(self.array <= other)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            if other._id == self._id:
                return Tensor(True)
            return Tensor(self.xp.array_equal(self.array == other.array))

        return Tensor(self.array == other)

    def __ne__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array != other.array)

        return Tensor(self.array != other)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array > other.array)

        return Tensor(self.array > other)

    def __ge__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array >= other.array)

        return Tensor(self.array >= other)

    def __repr__(self):
        name = f", name={self.name}" if self.name is not None else ""
        return f"Tensor({self.array.__str__()}{name})"

    def __getitem__(self, idx):
        return Tensor(self.array[idx], requires_grad=self.requires_grad)

    def __setitem__(self, idx, value):
        self.array[idx] = value

    @property
    def xp(self):
        """
        Returns the appropriate array library (numpy or cupy) for the tensor.

        Returns:
            module: The array library (numpy or cupy).
        """
        return select_backend(self.array)

    def einsum(self, subscript: str) -> "Tensor":
        """
        Performs an einsum operation on the tensor.

        Args:
            subscript (str): The einsum subscript string.

        Returns:
            Tensor: The result of the einsum operation.
        """
        from tricycle.einsum import Einsum

        return Einsum(subscript)(self)

    def repeat(self, n_repeats: int) -> "Tensor":
        """
        Repeats the tensor.

        Args:
            n_repeats (int): The number of times to repeat the tensor.

        Returns:
            Tensor: The repeated tensor.
        """
        from tricycle.ops import Repeat

        return Repeat()(self, n_repeats)

    @property
    def shape(self) -> Sequence[int]:
        """
        Returns the shape of the tensor.

        Returns:
            Sequence[int]: The shape of the tensor.
        """
        return self.array.shape

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the tensor.

        Returns:
            int: The number of dimensions.
        """
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        """
        Returns the data type of the tensor.

        Returns:
            np.dtype: The data type of the tensor.
        """
        return self.array.dtype

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        """
        Reshapes the tensor to the given shape.

        Args:
            shape (Sequence[int]): The new shape for the tensor.

        Returns:
            Tensor: The reshaped tensor.
        """
        from tricycle.ops import Reshape

        return Reshape()(self, shape)

    def split(self, n_splits: int, axis: int = -1) -> List["Tensor"]:
        """
        Splits the tensor into multiple sub-tensors.

        Args:
            n_splits (int): The number of splits to perform.
            axis (int, optional): The axis along which to split. Defaults to -1.

        Returns:
            List[Tensor]: A list of split tensors.
        """
        from tricycle.ops import Split

        return Split()(self, n_splits=n_splits, axis=axis)

    def mean(self) -> "Tensor":
        """
        Computes the mean of all elements in the tensor.

        Returns:
            Tensor: A new tensor containing the mean value.
        """
        from tricycle.ops import Mean

        return Mean()(self)

    def sum(self) -> "Tensor":
        """
        Computes the sum of all elements in the tensor.

        Returns:
            Tensor: A new tensor containing the sum.
        """
        from tricycle.unary import UnarySum

        return UnarySum()(self)

    def close_to(
        self,
        other: Union["Tensor", ArrayLike, float, int],
        equal_nan=False,
        rtol=1e-4,
        **kwargs,
    ) -> bool:
        """
        Checks if this tensor is close to another tensor or value within a tolerance.

        Args:
            other (Union[Tensor, ArrayLike, float, int]): The tensor or value to compare against.
            equal_nan (bool, optional): Whether to consider NaN values as equal. Defaults to False.
            rtol (float, optional): The relative tolerance parameter. Defaults to 1e-4.
            **kwargs: Additional keyword arguments to pass to numpy.allclose or cupy.allclose.

        Returns:
            bool: True if the tensors are close, False otherwise.
        """
        if not isinstance(other, Tensor):
            return self.xp.allclose(
                self.array,
                self.xp.array(other),
                equal_nan=equal_nan,
                rtol=rtol,
                **kwargs,
            )
        return self.xp.allclose(
            self.array, other.array, equal_nan=equal_nan, rtol=rtol, **kwargs
        )

    def to_batched(self):
        """
        Treats this tensor as a batch of tensors.

        Returns:
            Tensor: A new batched tensor.
        """
        from tricycle.unary import Batch

        return Batch()(self)

    def from_batched(self):
        """
        Treats a batched tensor as a normal, non-batched, tensor.

        Returns:
            Tensor: A new non-batched tensor.
        """
        from tricycle.unary import Unbatch

        return Unbatch()(self)

    @property
    def on_gpu(self):
        """
        Checks if the tensor is currently on the GPU.

        Returns:
            bool: True if the tensor is on the GPU, False otherwise.
        """
        if not GPU_ENABLED:
            return False
        import cupy

        return isinstance(self.array, cupy.ndarray)

    def to_gpu(self, device: int = 0):
        """
        Moves this tensor to the GPU, if cupy is enabled.

        Args:
            device (int, optional): The GPU device number. Defaults to 0.

        Returns:
            Tensor: The tensor moved to the GPU.

        Raises:
            GPUDisabledException: If CuPY is not enabled.
        """
        if not GPU_ENABLED:
            raise GPUDisabledException(
                "Cannot move tensor to GPU because CuPY is not enabled"
            )
        import cupy

        cupy.cuda.Device(device).use()
        self.array = cupy.asarray(self.array)
        return self

    def from_gpu(self):
        """
        Moves this tensor from the GPU to CPU.

        Returns:
            Tensor: The tensor moved to the CPU.

        Raises:
            GPUDisabledException: If CuPY is not enabled.
        """
        if not GPU_ENABLED:
            raise GPUDisabledException(
                "Cannot move tensor from GPU because CuPY is not enabled"
            )
        import cupy

        self.array = cupy.asnumpy(self.array)
        return self

    def zero_grad(self):
        """
        Removes any gradients or references to other tensors.

        Returns:
            Tensor: The tensor with gradients and references cleared.
        """
        self.grad = None
        self.args = None
        self.back_fns = None

        return self

    def numpy(self):
        """
        Returns the underlying array as a numpy array.

        Returns:
            np.ndarray: The tensor data as a numpy array.
        """
        if not GPU_ENABLED:
            return self.array

        import cupy

        return cupy.asnumpy(self.array) if self.on_gpu else self.array


def select_backend(*tensors: Tensor | np.ndarray | ArrayLike):
    """
    Given some tensors, if any of them are on the GPU, return the cupy
    backend. Otherwise default to the numpy backend.

    Args:
        *tensors: Variable number of tensors or arrays to check.

    Returns:
        module: The appropriate backend module (numpy or cupy).
    """
    if not GPU_ENABLED:
        return np

    import cupy

    return cupy.get_array_module(*tensors)
