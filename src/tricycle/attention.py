"""Attention module for multi-head attention operations.

This module implements the multi-head attention mechanism as described in
"Attention Is All You Need" (Vaswani et al., 2017). It includes functions for
building attention masks and the main Attention class for performing
multi-head attention operations.
"""

import ctypes
import math
from math import sqrt
from pathlib import Path

import cupy as cp
import numpy as np

from tricycle import GPU_ENABLED
from tricycle.context import TRICYCLE_CONTEXT
from tricycle.exceptions import GPUDisabledException
from tricycle.kernels import _attn_bwd, _attn_fwd
from tricycle.ops import Op
from tricycle.tensor import Tensor


def build_mask(context_window: int, n_heads: int) -> Tensor:
    """Build an attention mask to prevent attending to future tokens.

    This function creates a boolean mask that can be used in multi-head attention
    mechanisms to implement causal (unidirectional) attention.

    Args:
        context_window: An integer representing the size of the context window.
        n_heads: An integer representing the number of attention heads.

    Returns:
        A boolean tensor of shape (n_heads, context_window, context_window)
        representing the attention mask.
    """
    mask = np.ones((context_window, context_window), dtype=bool)
    idx = np.tril(mask)
    mask = np.stack([~idx] * n_heads)
    return mask


class Attention(Op):
    """Multi-head attention operation.

    This class implements the multi-head attention mechanism as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Attributes:
        embedding_dim: An integer representing the dimension of the input embeddings.
        n_heads: An integer representing the number of attention heads.
        context_window: An integer representing the size of the context window.
        mask: A tensor representing the attention mask.
        _grad: A tensor to store gradients during backpropagation.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
    ):
        """Initialize the Attention operation.

        Args:
            embedding_dim: An integer representing the dimension of the input embeddings.
            n_heads: An integer representing the number of attention heads.
            context_window: An integer representing the size of the context window.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.context_window = context_window
        self.mask = build_mask(
            context_window=self.context_window, n_heads=self.n_heads
        )
        self._grad = None

    def backward(self, grad: Tensor):
        """Compute the gradient of the attention operation.

        Args:
            grad: A Tensor representing the upstream gradient.

        Returns:
            A Tensor representing the gradient with respect to the input.
        """
        xp = grad.xp
        in_shape = (self.batch_size, self.context_window, self.embedding_dim)

        attention = grad.array

        # TODO: come up with a better name
        # smush
        attention = attention.reshape(
            (
                self.batch_size,
                self.context_window,
                self.n_heads,
                self.head_size,
            )
        )
        value = xp.einsum("BNIj, BINH -> BNjH", self._before_smush, attention)
        attention = xp.einsum("BINH, BNjH -> BNIj", attention, self._value)

        # softmax
        inner = xp.sum(attention * self._before_smush, axis=-1, keepdims=True)
        attention = self._before_smush * (attention - inner)

        # mask
        attention = xp.where(
            self.mask[:, : self.n_tokens, : self.n_tokens], 0, attention
        )
        attention /= self.divisor

        # attend
        query = xp.einsum("BNIJ, BNJh -> BNIh", attention, self._key)
        key = xp.einsum("BNIh, BNIJ -> BNJh", self._query, attention)

        # reshape + reorder
        key = xp.einsum("BNTH->BTNH", key)
        query = xp.einsum("BNTH->BTNH", query)
        value = xp.einsum("BNTH->BTNH", value)

        key = key.reshape(in_shape)
        query = query.reshape(in_shape)
        value = value.reshape(in_shape)

        # merge into single tensor
        if self._grad is None:
            self._grad = xp.zeros(
                (
                    self.batch_size,
                    self.context_window,
                    self.embedding_dim * 3,
                )
            )
        self._grad[:, :, : self.embedding_dim] = query
        self._grad[:, :, self.embedding_dim : self.embedding_dim * 2] = key
        self._grad[:, :, self.embedding_dim * 2 :] = value

        return Tensor(self._grad)

    def forward(self, tensor: Tensor):
        """Apply the multi-head attention operation to the input tensor.

        Args:
            tensor: A Tensor of shape (batch_size, seq_len, embedding_dim * 3).
                The input should contain concatenated query, key, and value projections.

        Returns:
            A Tensor representing the output after applying multi-head attention.
        """
        xp = tensor.xp

        assert tensor.is_batched

        # split the input into 3 peices
        self._input = tensor
        query = tensor[:, :, : self.embedding_dim]
        key = tensor[:, :, self.embedding_dim : self.embedding_dim * 2]
        value = tensor[:, :, self.embedding_dim * 2 :]

        # Figure out how big everything is
        self.batch_size = key.array.shape[0]
        self.head_size = self.embedding_dim // self.n_heads
        self.n_tokens = key.shape[-2]
        head_shape = (
            self.batch_size,
            self.n_tokens,  # number of tokens
            self.n_heads,  # number of heads
            self.head_size,  # embedding per head
        )
        out_shape = (self.batch_size, self.n_tokens, self.embedding_dim)

        # reshape and reorder the heads
        key = key.array
        query = query.array
        value = value.array

        key = key.reshape(head_shape)
        query = query.reshape(head_shape)
        value = value.reshape(head_shape)

        key = xp.einsum("BTNH->BNTH", key)
        query = xp.einsum("BTNH->BNTH", query)
        value = xp.einsum("BTNH->BNTH", value)

        self._key = key
        self._query = query
        self._value = value

        # attend
        self.divisor = sqrt(self.head_size)
        attention = xp.einsum("BNIh, BNJh -> BNIJ", query, key)
        attention = attention / self.divisor

        # mask
        attention = xp.where(
            self.mask[:, : self.n_tokens, : self.n_tokens], -xp.inf, attention
        )

        # Exponents tend to overflow/underflow when using 16 bit precision
        # so we need to switch to 32 bit
        if TRICYCLE_CONTEXT.use_mixed_precision:
            attention = attention.astype(xp.float32)

        # softmax
        exp = xp.exp(attention - xp.max(attention, axis=-1, keepdims=True))
        denominator = xp.sum(exp, axis=-1, keepdims=True)
        attention = exp / denominator

        if TRICYCLE_CONTEXT.use_mixed_precision:
            attention = attention.astype(xp.float16)

        # smush the heads back together
        self._before_smush = attention
        attention = xp.einsum("BNTi, BNiH -> BTNH", attention, value)
        attention = attention.reshape(out_shape)

        result = Tensor(attention, is_batched=True)
        result.back_fns = (self.backward,)
        result.args = (self._input,)
        return result

    def to_gpu(self, device: int = 0):
        """Move this operation to a GPU.

        Args:
            device: An integer representing the GPU device number.
        """
        if GPU_ENABLED:
            import cupy as cp

            cp.cuda.Device(device).use()
            self.mask = cp.array(self.mask)

    def from_gpu(self):
        """Move the operation back to CPU."""
        if GPU_ENABLED:
            import cupy as cp

            self.mask = cp.asnumpy(self.mask)


class CudaAttention(Op):
    """Multi-head attention operation with handwritten cuda kernel.

    This class implements the multi-head attention mechanism as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Attributes:
        embedding_dim: An integer representing the dimension of the input embeddings.
        n_heads: An integer representing the number of attention heads.
        context_window: An integer representing the size of the context window.
        mask: A tensor representing the attention mask.
        _grad: A tensor to store gradients during backpropagation.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
    ):
        """Initialize the Attention operation.

        Args:
            embedding_dim: An integer representing the dimension of the input embeddings.
            n_heads: An integer representing the number of attention heads.
            context_window: An integer representing the size of the context window.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.context_window = context_window
        self.mask = build_mask(
            context_window=self.context_window, n_heads=self.n_heads
        )
        self._grad = None

    def backward(self, grad: Tensor):
        """Compute the gradient of the attention operation.

        Args:
            grad: A Tensor representing the upstream gradient.

        Returns:
            A Tensor representing the gradient with respect to the input.
        """
        xp = grad.xp
        in_shape = (self.batch_size, self.context_window, self.embedding_dim)

        attention = grad.array

        # TODO: come up with a better name
        # smush
        attention = attention.reshape(
            (
                self.batch_size,
                self.context_window,
                self.n_heads,
                self.head_size,
            )
        )
        value = xp.einsum("BNIj, BINH -> BNjH", self._before_smush, attention)
        attention = xp.einsum("BINH, BNjH -> BNIj", attention, self._value)

        # softmax
        inner = xp.sum(attention * self._before_smush, axis=-1, keepdims=True)
        attention = self._before_smush * (attention - inner)

        # mask
        attention = xp.where(
            self.mask[:, : self.n_tokens, : self.n_tokens], 0, attention
        )
        attention /= self.divisor

        # attend
        query = xp.einsum("BNIJ, BNJh -> BNIh", attention, self._key)
        key = xp.einsum("BNIh, BNIJ -> BNJh", self._query, attention)

        # reshape + reorder
        key = xp.einsum("BNTH->BTNH", key)
        query = xp.einsum("BNTH->BTNH", query)
        value = xp.einsum("BNTH->BTNH", value)

        key = key.reshape(in_shape)
        query = query.reshape(in_shape)
        value = value.reshape(in_shape)

        # merge into single tensor
        if self._grad is None:
            self._grad = xp.zeros(
                (
                    self.batch_size,
                    self.context_window,
                    self.embedding_dim * 3,
                )
            )
        self._grad[:, :, : self.embedding_dim] = query
        self._grad[:, :, self.embedding_dim : self.embedding_dim * 2] = key
        self._grad[:, :, self.embedding_dim * 2 :] = value

        return Tensor(self._grad)

    def forward(self, tensor: Tensor):
        """Apply the multi-head attention operation to the input tensor.

        Args:
            tensor: A Tensor of shape (batch_size, seq_len, embedding_dim * 3).
                The input should contain concatenated query, key, and value projections.

        Returns:
            A Tensor representing the output after applying multi-head attention.
        """
        xp = tensor.xp

        assert tensor.is_batched

        # split the input into 3 peices
        self._input = tensor
        query = tensor[:, :, : self.embedding_dim]
        key = tensor[:, :, self.embedding_dim : self.embedding_dim * 2]
        value = tensor[:, :, self.embedding_dim * 2 :]

        # Figure out how big everything is
        self.batch_size = key.array.shape[0]
        self.head_size = self.embedding_dim // self.n_heads
        self.n_tokens = key.shape[-2]
        head_shape = (
            self.batch_size,
            self.n_tokens,  # number of tokens
            self.n_heads,  # number of heads
            self.head_size,  # embedding per head
        )
        out_shape = (self.batch_size, self.n_tokens, self.embedding_dim)

        # reshape and reorder the heads
        key = key.array
        query = query.array
        value = value.array

        key = key.reshape(head_shape)
        query = query.reshape(head_shape)
        value = value.reshape(head_shape)

        key = xp.einsum("BTNH->BNTH", key)
        query = xp.einsum("BTNH->BNTH", query)
        value = xp.einsum("BTNH->BNTH", value)

        self._key = key
        self._query = query
        self._value = value

        # attend
        self.divisor = sqrt(self.head_size)
        attention = xp.einsum("BNIh, BNJh -> BNIJ", query, key)
        attention = attention / self.divisor

        # mask
        attention = xp.where(
            self.mask[:, : self.n_tokens, : self.n_tokens], -xp.inf, attention
        )

        # Exponents tend to overflow/underflow when using 16 bit precision
        # so we need to switch to 32 bit
        if TRICYCLE_CONTEXT.use_mixed_precision:
            attention = attention.astype(xp.float32)

        # softmax
        exp = xp.exp(attention - xp.max(attention, axis=-1, keepdims=True))
        denominator = xp.sum(exp, axis=-1, keepdims=True)
        attention = exp / denominator

        if TRICYCLE_CONTEXT.use_mixed_precision:
            attention = attention.astype(xp.float16)

        # smush the heads back together
        self._before_smush = attention
        attention = xp.einsum("BNTi, BNiH -> BTNH", attention, value)
        attention = attention.reshape(out_shape)

        result = Tensor(attention, is_batched=True)
        result.back_fns = (self.backward,)
        result.args = (self._input,)
        return result

    def to_gpu(self, device: int = 0):
        """Move this operation to a GPU.

        Args:
            device: An integer representing the GPU device number.
        """
        if GPU_ENABLED:
            import cupy as cp

            cp.cuda.Device(device).use()
            self.mask = cp.array(self.mask)

    def from_gpu(self):
        """Move the operation back to CPU."""
        if GPU_ENABLED:
            import cupy as cp

            self.mask = cp.asnumpy(self.mask)


# CudnnAttention class has been removed


class TritonAttention(Op):
    batch_size: int
    n_tokens: int
    embedding_dim: int
    n_heads: int
    causal: bool = True
    sm_scale: float = 0.5

    def __init__(
        self,
        batch_size: int,
        n_tokens: int,
        embedding_dim: int,
        n_heads: int,
    ):
        if not GPU_ENABLED:
            raise GPUDisabledException("Cannot use Triton without a GPU")

        # sizes
        self.batch_size = batch_size
        self.n_tokens = n_tokens
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        if self.embedding_dim % self.n_heads != 0:
            raise ValueError(
                f"n_heads must divide embedding dim. Found {n_heads=}, {embedding_dim=}"
            )
        self.head_size = self.embedding_dim // self.n_heads

        self.result = None
        self.mask = None
        self.q = None
        self.k = None
        self.v = None

    def _fwd_grid(self, args):
        import triton

        return (
            triton.cdiv(self.n_tokens, args["BLOCK_M"]),
            self.batch_size * self.n_heads,
            1,
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Attention with a custom flash-attention triton kernel
        """
        if tensor.xp is not cp:
            raise ValueError("Cannot use numpy arrays with Triton")
        xp = tensor.xp

        self._input = tensor

        # Set scale factor based on head size
        self.sm_scale = 1.0 / math.sqrt(self.head_size)

        # q,k and v start in the same array so we need to extract them out to contiguous arrays
        # for separate processing
        embedding_dim = tensor.array.shape[-1] // 3
        q = tensor.array[:, :, :embedding_dim]
        k = tensor.array[:, :, embedding_dim : 2 * embedding_dim]
        v = tensor.array[:, :, 2 * embedding_dim :]

        q = xp.ascontiguousarray(q)
        k = xp.ascontiguousarray(k)
        v = xp.ascontiguousarray(v)
        # Reshaping and transposing
        head_shape = (
            self.batch_size,
            self.n_tokens,
            self.n_heads,
            self.head_size,
        )
        k = k.reshape(*head_shape).transpose(0, 2, 1, 3)
        q = q.reshape(*head_shape).transpose(0, 2, 1, 3)
        v = v.reshape(*head_shape).transpose(0, 2, 1, 3)

        self.k = Tensor(k, dtype=k.dtype)
        self.q = Tensor(q, dtype=k.dtype)
        self.v = Tensor(v, dtype=k.dtype)

        # check that triton will be happy with head_size
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = (
            q.shape[-1],
            k.shape[-1],
            v.shape[-1],
        )
        assert HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V == self.head_size
        assert self.head_size in {16, 32, 64, 128, 256}

        # allocate memory for output
        if self.result is None:
            self.result = Tensor(
                xp.empty_like(self.q.array),
                dtype=tensor.array.dtype,
            ).to_gpu()

        # allocate memory for mask
        if self.mask is None:
            self.mask = Tensor(
                xp.zeros(
                    (self.batch_size, self.n_heads, self.n_tokens),
                ),
                dtype=xp.float32,
            ).to_gpu()

        stage = 3 if self.causal else 1
        extra_kern_args = {}

        _attn_fwd[self._fwd_grid](
            self.q,
            self.k,
            self.v,
            self.sm_scale,
            self.mask,
            self.result,
            self.q.strides[0],
            self.q.strides[1],
            self.q.strides[2],
            self.q.strides[3],
            self.k.strides[0],
            self.k.strides[1],
            self.k.strides[2],
            self.k.strides[3],
            self.v.strides[0],
            self.v.strides[1],
            self.v.strides[2],
            self.v.strides[3],
            self.result.strides[0],
            self.result.strides[1],
            self.result.strides[2],
            self.result.strides[3],
            self.batch_size,
            self.n_heads,
            N_CTX=self.n_tokens,
            HEAD_DIM=self.head_size,
            STAGE=stage,
            **extra_kern_args,
        )

        # recombine into (batch_size, n_tokens, embedding_dim)
        return Tensor(
            self.result.array.transpose(0, 2, 1, 3).reshape(
                self.batch_size, self.n_tokens, self.embedding_dim
            ),
            is_batched=True,
            args=(self._input,),
            back_fns=(self.backward,),
        )

    def backward(self, grad: Tensor):
        """
        Attention with a custom cuda kernel
        """
        if grad.xp is not cp:
            raise ValueError("Cannot use numpy arrays with Triton")
        self.output_grad = grad

        assert (
            self.q.stride()
            == self.k.stride()
            == self.v.stride()
            == self.result.stride()
            == grad.stride()
        )

    def to_gpu(self, *_):
        pass
