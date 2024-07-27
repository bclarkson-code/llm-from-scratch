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

from tricycle import GPU_ENABLED, TRICYCLE_CONTEXT
from tricycle.exceptions import GPUDisabledException
from tricycle.weakset import WeakSet

if TYPE_CHECKING:
    from tricycle.ops import Op

logger = logging.getLogger(__name__)

DEFAULT_DTYPE = np.float32


class Tensor:
    """
    An N-dimensional grid of numbers. This is implemented as a subclass
    of a standard numpy array
    """

    _id: int
    array: ArrayLike
    args: tuple["Tensor", ...] | None = None
    back_fns: tuple["Op", ...] | None = None
    parents: set["Tensor"] | None = None
    grad: Optional["Tensor"] = None
    name: Optional[str] = None
    requires_grad: bool = True
    is_batched: bool = False

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
        if isinstance(array, Tensor):
            self = array
            return
        self._id = _id or uuid.uuid4().int
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
        if self.array.dtype != dtype:
            self.array = self.array.astype(dtype)

        self.requires_grad = requires_grad
        self.is_batched = is_batched
        self.args = args
        self.back_fns = back_fns
        self.name = name

    def _attach_parents(self):
        """
        Traverse through the graph, labelling each tensor with the tensors that
        are direct parents to it in the graph.

        We're doing this so that we can traverse through the graph later in
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

                if arg.parents is None:
                    # if we use a set, we get a circular reference
                    # which can't be garbage collected, leading to a memory
                    # leak so we need to do a weakref to avoid the circular
                    # reference
                    arg.parents = WeakSet()

                # if a node has a parent we haven't visited yet, store it
                if node not in arg.parents:
                    stack.append(arg)
                    arg.parents.add(node)

    def _calculate_gradients(self, clip: float | None = None):
        """
        Because every output of an `Op` stores the inputs that were used to
        make it, we can think of the outputs of `Op`s as a tree of
        intermediate values where the final output of a network is the root
        node and the inputs are leaves.

        Thanks to the chain rule, we can calculate the derivative of the
        output wrt an input by moving from the output (root node) to the
        input, applying each back_fn we go through to get there.

        It turns out that we can minimise calculations by only visiting a
        child node if all of its parents have been visited through every
        possible path: a topological sort.
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

                if arg.parents is None:
                    raise ValueError(
                        "arg.parents is None. Parents must be attached",
                        "before calculating gradients. Did you forget to ",
                        "call _attach_parents?",
                    )

                # already visited along this edge, dont do it again
                if node not in arg.parents:
                    continue

                arg.parents.remove(node)

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
                if len(arg.parents) == 0:
                    # get rid of the weakref once we're done with a node so we
                    # can pickle the model. Weakrefs can't be pickled
                    arg.parents = None
                    stack.append(arg)

    def backward(self, clip: float | None = None):
        """
        Perform a backward pass through the graph, calculating the gradient
        for each parameter
        """
        self._attach_parents()
        self._calculate_gradients(clip=clip)

    def __hash__(self) -> int:
        return self._id

    def __add__(self, other: Union[float, "Tensor"]) -> "Tensor":
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
        return select_backend(self.array)

    def einsum(self, subscript: str) -> "Tensor":
        """
        Perform an einsum operation on the tensor
        """
        from tricycle.einsum import Einsum

        return Einsum(subscript)(self)

    def repeat(self, n_repeats: int) -> "Tensor":
        from tricycle.ops import Repeat

        return Repeat()(self, n_repeats)

    @property
    def shape(self) -> Sequence[int]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        from tricycle.ops import Reshape

        return Reshape()(self, shape)

    def split(self, n_splits: int, axis: int = -1) -> List["Tensor"]:
        from tricycle.ops import Split

        return Split()(self, n_splits=n_splits, axis=axis)

    def mean(self) -> "Tensor":
        from tricycle.ops import Mean

        return Mean()(self)

    def sum(self) -> "Tensor":
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
        Convenience method to check if two tensors are identical
        to within some tolerance
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
        Treat this tensor as a batch of tensors
        """
        from tricycle.unary import Batch

        return Batch()(self)

    def from_batched(self):
        """
        Treat a batched tensor as a normal, non-batched, tensor
        """
        from tricycle.unary import Unbatch

        return Unbatch()(self)

    @property
    def on_gpu(self):
        if not GPU_ENABLED:
            return False
        import cupy

        return isinstance(self.array, cupy.ndarray)

    def to_gpu(self, device: int = 0):
        """
        Move this tensor to the GPU, if cupy is enabled
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
        Move this tensor from the GPU to CPU
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
        Remove any gradients or references to other tensors
        """
        self.grad = None
        self.args = None
        self.back_fns = None

        return self

    def numpy(self):
        """
        Return the underlying array as a numpy array
        """
        if not GPU_ENABLED:
            return self.array

        import cupy

        return cupy.asnumpy(self.array) if self.on_gpu else self.array


def select_backend(*tensors: Tensor | np.ndarray | ArrayLike):
    """
    Given some tensors, if any of them are on the GPU, return the cupy
    backend. Otherwise default to the numpy backend
    """
    if not GPU_ENABLED:
        return np

    import cupy

    return cupy.get_array_module(*tensors)
