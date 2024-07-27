"""Provides reduction operations for tensors.

This module contains classes for performing max and min reduction operations
on tensors using einsum notation.
"""

from tricycle.einsum import Einsum, Subscript
from tricycle.ops import Op
from tricycle.tensor import Tensor


class ReduceMax(Op):
    """Performs max reduction on a tensor along specified dimensions."""

    def __call__(self, tensor: Tensor, subscript: Subscript | str):
        """Generates an indicator tensor for max reduction using einsum.

        This method creates an indicator tensor that, when einsummed with the
        input tensor, results in a tensor equal to the max applied along the
        indices that don't appear in the output of the subscript.

        Args:
            tensor: The input tensor to perform max reduction on.
            subscript: The einsum subscript specifying the reduction.

        Returns:
            A Tensor representing the result of the max reduction.

        Raises:
            AssertionError: If the subscript suggests more than one input tensor.
        """
        if isinstance(subscript, str):
            subscript = Subscript(subscript)

        assert (
            len(subscript.inputs) == 1
        ), f"Can only reduce a single tensor at a time. Indices suggeststed: {len(subscript.inputs)} tensors: {subscript.inputs}"

        [idx] = subscript.inputs

        reduce_along_axes = [
            i for i, char in enumerate(idx) if char not in subscript.output
        ]

        if not reduce_along_axes:
            return tensor

        indicator = tensor.array == tensor.xp.max(
            tensor.array, axis=tuple(reduce_along_axes), keepdims=True
        )
        indicator = Tensor(
            indicator, requires_grad=False, is_batched=tensor.is_batched
        )
        indicator.array = indicator.array.astype(tensor.xp.int8)

        new_subscript = Subscript.from_split([idx, idx], subscript.output)

        result = Einsum(new_subscript)(tensor, indicator)
        result.name = f"min({new_subscript})"

        return result


class ReduceMin(Op):
    """Performs min reduction on a tensor along specified dimensions."""

    def __call__(self, tensor: Tensor, subscript: Subscript | str):
        """Generates an indicator tensor for min reduction using einsum.

        This method creates an indicator tensor that, when einsummed with the
        input tensor, results in a tensor equal to the min applied along the
        indices that don't appear in the output of the subscript.

        Args:
            tensor: The input tensor to perform min reduction on.
            subscript: The einsum subscript specifying the reduction.

        Returns:
            A Tensor representing the result of the min reduction.

        Raises:
            AssertionError: If the subscript suggests more than one input tensor.
        """
        if isinstance(subscript, str):
            subscript = Subscript(subscript)

        assert (
            len(subscript.inputs) == 1
        ), f"Can only reduce a single tensor at a time. Indices suggeststed: {len(subscript.inputs)} tensors: {subscript.inputs}"

        [idx] = subscript.inputs

        reduce_along_axes = [
            i for i, char in enumerate(idx) if char not in subscript.output
        ]

        if not reduce_along_axes:
            return tensor

        indicator = tensor.array == tensor.xp.min(
            tensor.array, axis=tuple(reduce_along_axes), keepdims=True
        )
        indicator = Tensor(
            indicator, requires_grad=False, is_batched=tensor.is_batched
        )
        indicator.array = indicator.array.astype(tensor.xp.int8)

        new_subscript = Subscript.from_split([idx, idx], subscript.output)

        result = Einsum(new_subscript)(tensor, indicator)
        result.name = f"min({new_subscript})"

        return result
