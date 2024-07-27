"""
Einsum implementation for generalized matrix operations.

This module provides an implementation of the einsum operation, which is a
generalization of many matrix operations. It allows for flexible manipulation
of tensors using index notation.

Example usage:
    >>> a = Tensor([[1,2],[3,4]])
    >>> Einsum("ij->ji")(a)
    Tensor([[1. 3.]
     [2. 4.]], name=einsum ij->ji)

For more details on einsum operations, refer to the class and function
docstrings.
"""

import itertools
import re
from typing import Sequence

from tricycle.tensor import Tensor, select_backend


class Subscript:
    """
    A string that defines an einsum operation.

    This class parses and represents the subscript notation used in einsum
    operations.

    Attributes:
        subscript (str): The original subscript string.
        inputs (list[list[str]]): Parsed input indices.
        output (list[str]): Parsed output indices.

    """

    subscript: str
    inputs: list[list[str]]
    output: list[str]
    _index_pattern: re.Pattern = re.compile(r"(?:[A-Za-z]|(?:\.{3}))")

    def __init__(self, subscript: str):
        """
        Initialize a Subscript object.

        Args:
            subscript (str): The einsum subscript string.
        """
        self.subscript = subscript
        self.inputs, self.output = self.split()

    def split(self) -> tuple[list[list[str]], list[str]]:
        """
        Parse a subscripts string into a list of indices and a result.

        Returns:
            tuple: A tuple containing two elements:
                - list[list[str]]: Parsed input indices.
                - list[str]: Parsed output indices.
        """
        indices, result = self.subscript.split("->")
        indices = indices.split(",")
        indices = [re.findall(self._index_pattern, idx) for idx in indices]
        result = re.findall(self._index_pattern, result)
        return indices, result

    @staticmethod
    def join(indices: list[list[str]], result: list[str]) -> str:
        """
        Join parsed indices and result back into a subscript string.

        Args:
            indices (list[list[str]]): Parsed input indices.
            result (list[str]): Parsed output indices.

        Returns:
            str: The joined subscript string.
        """
        inputs_string = ",".join(["".join(idx) for idx in indices])
        outputs_string = "".join(result)
        return f"{inputs_string}->{outputs_string}"

    @classmethod
    def from_split(cls, indices: list[list[str]], result: list[str]):
        """
        Create a Subscript object from split indices and result.

        Args:
            indices (list[list[str]]): Parsed input indices.
            result (list[str]): Parsed output indices.

        Returns:
            Subscript: A new Subscript object.
        """
        return cls(cls.join(indices, result))

    @property
    def unique_input_indices(self) -> set[str]:
        """
        Get the set of unique input indices.

        Returns:
            set[str]: Set of unique input indices.
        """
        all_inputs = itertools.chain(*self.inputs)
        return set(all_inputs)

    def __repr__(self):
        """
        Return a string representation of the Subscript object.

        Returns:
            str: The subscript string.
        """
        return self.subscript

    def __str__(self):
        """
        Return a string representation of the Subscript object.

        Returns:
            str: The subscript string.
        """
        return self.subscript


class EinsumBackOp:
    """
    The backward operation for an einsum operation.

    This class represents the backward pass of an einsum operation, which is
    done by swapping the indices and tensors for an input with the output.

    Attributes:
        idx (int): The index of the input tensor for which this backward
            operation is defined.
        tensors (Sequence[Tensor]): The input tensors of the original einsum
            operation.
        subscript (Subscript): The subscript of the original einsum operation.
        left_tensors (Sequence[Tensor]): Tensors to the left of the current
            input in the original operation.
        right_tensors (Sequence[Tensor]): Tensors to the right of the current
            input in the original operation.
        combined_subscript (Subscript): The subscript for the backward
            operation.
    """

    def __init__(
        self, idx: int, tensors: Sequence[Tensor], subscript: Subscript
    ):
        """
        Initialize an EinsumBackOp object.

        Args:
            idx (int): The index of the input tensor for which this backward
                operation is defined.
            tensors (Sequence[Tensor]): The input tensors of the original
                einsum operation.
            subscript (Subscript): The subscript of the original einsum
                operation.
        """
        self.idx = idx
        self.tensors = tensors
        self.subscript = subscript

        self.left_tensors, self.right_tensors = self._build_inputs()
        self.combined_subscript = self._build_subscript()

    def _build_inputs(self):
        """
        Build the left and right tensor sequences for the backward operation.

        Returns:
            tuple: A tuple containing two elements:
                - Sequence[Tensor]: Left tensors.
                - Sequence[Tensor]: Right tensors.
        """
        left_tensors = self.tensors[: self.idx]

        # Special case for the last index
        if self.idx < len(self.tensors) - 1:
            right_tensors = self.tensors[self.idx + 1 :]
        else:
            right_tensors = []

        return left_tensors, right_tensors

    def _build_subscript(self):
        """
        Build the subscript for the backward operation.

        Returns:
            Subscript: The subscript for the backward operation.
        """
        left_subscript = self.subscript.inputs[: self.idx]

        # Special case for the last index
        if self.idx < len(self.tensors) - 1:
            right_subscript = self.subscript.inputs[self.idx + 1 :]
        else:
            right_subscript = []

        combined_indices = [
            *left_subscript,
            self.subscript.output,
            *right_subscript,
        ]
        return Subscript.from_split(
            combined_indices, self.subscript.inputs[self.idx]
        )

    def __call__(self, tensor: Tensor):
        """
        Build the backward function for einsum.

        This is done by swapping the indices and tensors for an input with
        the output. E.g "ij,jk->ik" with idx = 0 would become "ik,jk->ij"

        Args:
            tensor (Tensor): The gradient tensor from the previous layer.

        Returns:
            Tensor: The result of the backward einsum operation.
        """
        combined_tensors = [*self.left_tensors, tensor, *self.right_tensors]
        return Einsum(self.combined_subscript)(*combined_tensors)

    def __repr__(self):
        """
        Return a string representation of the EinsumBackOp object.

        Returns:
            str: A string representation of the object.
        """
        return f"EinsumBackOp({self.combined_subscript})"


class Einsum:
    """
    A class representing an einsum operation.

    This class encapsulates the logic for performing einsum operations on
    tensors, including handling of batched operations and backward passes.

    Attributes:
        subscript (Subscript): The subscript defining the einsum operation.
    """

    subscript: Subscript

    def __init__(self, subscript: str | Subscript):
        """
        Initialize an Einsum object.

        Args:
            subscript (str | Subscript): The subscript defining the einsum
                operation. Can be a string or a Subscript object.
        """
        if isinstance(subscript, str):
            subscript = Subscript(subscript)
        self.subscript = subscript

    def _build_back_ops(self, tensors: Sequence[Tensor], subscript: Subscript):
        """
        Figure out the backward operation for each input.

        Args:
            tensors (Sequence[Tensor]): The input tensors.
            subscript (Subscript): The subscript for the operation.

        Returns:
            list: A list of EinsumBackOp objects, one for each input tensor.
        """
        assert len(tensors) == len(subscript.inputs)

        # To avoid adding a bunch of special cases for batched
        # operations, we replace any batched operations with
        # their non-batched counterparts
        subscript = Subscript(subscript.subscript.replace("z", ""))

        back_functions = []
        for idx in range(len(tensors)):
            back_op = EinsumBackOp(idx, tensors, subscript)
            back_functions.append(back_op)

        return back_functions

    def _handle_single_tensor(
        self, subscript: Subscript, tensors: Sequence[Tensor]
    ) -> tuple[Subscript, Sequence[Tensor]]:
        """
        Handle the case of a single input tensor.

        If there is only one tensor, we need to insert a matrix of ones
        to allow for expansion operations.

        Args:
            subscript (Subscript): The original subscript.
            tensors (Sequence[Tensor]): The input tensors.

        Returns:
            tuple: A tuple containing two elements:
                - Subscript: The modified subscript.
                - Sequence[Tensor]: The modified list of tensors.
        """
        xp = select_backend(*tensors)
        if len(tensors) != 1:
            return subscript, tensors

        [tensor] = tensors
        ones = Tensor(
            xp.ones(tensor.shape),
            is_batched=tensor.is_batched,
            requires_grad=False,
        )
        tensors = [tensor, ones]

        [index] = subscript.inputs
        output = subscript.output
        inputs = [index, index]
        subscript = Subscript.from_split(inputs, output)

        return subscript, tensors

    def _handle_batched(
        self, subscript: Subscript, tensors: Sequence[Tensor]
    ) -> tuple[Subscript, Sequence[Tensor], bool]:
        """
        Handle batched tensors in the einsum operation.

        If a tensor is labelled as being batched, add an extra dimension
        to its indices.

        Args:
            subscript (Subscript): The original subscript.
            tensors (Sequence[Tensor]): The input tensors.

        Returns:
            tuple: A tuple containing three elements:
                - Subscript: The modified subscript.
                - Sequence[Tensor]: The input tensors (unchanged).
                - bool: Whether the output should be batched.

        Raises:
            ValueError: If 'z' is used in the subscript for non-batched tensors.
        """
        inputs = []
        batch_output = False
        for idx, tensor in zip(subscript.inputs, tensors):
            if tensor.is_batched:
                inputs.append(["z"] + idx)
                batch_output = True
            else:
                inputs.append(idx)
        output = subscript.output
        if batch_output:
            if "z" in subscript.subscript:
                raise ValueError(
                    "`z` cannot be used in an einsum subscript on "
                    "non-batched tensors because "
                    "it is reserved for batched indices."
                )
            output = ["z"] + output

        subscript = Subscript.from_split(inputs, output)
        return subscript, tensors, batch_output

    def _replace_infinity(self, tensors: Sequence[Tensor]):
        """
        Replace infinity values in tensors with the max value for that datatype.

        Args:
            tensors (Sequence[Tensor]): The input tensors.

        Returns:
            list: A list of processed tensors with infinity values replaced.
        """
        xp = select_backend(*tensors)
        processed = []
        for tensor in tensors:
            if not xp.isinf(tensor.array).any():
                processed.append(tensor)
                continue

            new_tensor = Tensor(
                xp.nan_to_num(tensor.array),
                is_batched=tensor.is_batched,
            )
            new_tensor.args = tensor.args
            new_tensor.back_fns = tensor.back_fns
            new_tensor.name = tensor.name
            processed.append(new_tensor)

        return processed

    def __call__(self, *tensors: Tensor):
        """
        Perform the einsum operation on the input tensors.

        Args:
            *tensors (Tensor): The input tensors for the einsum operation.

        Returns:
            Tensor: The result of the einsum operation.
        """
        xp = select_backend(*tensors)
        subscript, tensors, batch_output = self._handle_batched(
            self.subscript, tensors
        )
        subscript, tensors = self._handle_single_tensor(subscript, tensors)
        tensor_data = [t.array for t in tensors]
        result = Tensor(xp.einsum(str(subscript), *tensor_data))
        if batch_output:
            result.is_batched = True

        result.args = tuple(tensors)
        result.back_fns = tuple(self._build_back_ops(tensors, subscript))
        result.name = f"einsum {self.subscript}"
        return result
