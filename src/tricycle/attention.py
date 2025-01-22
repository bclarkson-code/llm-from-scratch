"""Attention module for multi-head attention operations.

This module implements the multi-head attention mechanism as described in
"Attention Is All You Need" (Vaswani et al., 2017). It includes functions for
building attention masks and the main Attention class for performing
multi-head attention operations.
"""

import ctypes
from math import sqrt
from pathlib import Path

import cupy as cp
import numpy as np

from tricycle import GPU_ENABLED
from tricycle.context import TRICYCLE_CONTEXT
from tricycle.exceptions import GPUDisabledException
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


class CudnnAttention(Op):
    lib_path: Path = (
        Path(__file__).parent.parent.parent / "build/attention_cudnn.so"
    )
    batch_size: int
    context_window: int
    embedding_dim: int
    n_heads: int

    def __init__(
        self,
        batch_size: int,
        context_window: int,
        embedding_dim: int,
        n_heads: int,
        shared: dict[str, cp.ndarray],
    ):
        if not GPU_ENABLED:
            raise GPUDisabledException("Cannot use CUDNN without a GPU")
        # if not TRICYCLE_CONTEXT.use_mixed_precision:
        #     raise NotImplementedError(
        #         "CUDNN attention is only supported with FP16"
        #     )
        self.batch_size = batch_size
        self.context_window = context_window
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.shared = shared

        self.initialise_kernels()

        input_shape = (
            self.batch_size,
            self.context_window,
            3 * self.embedding_dim,
        )
        self.input = cp.zeros(input_shape, dtype=cp.float16)

        stats_shape = (self.batch_size, self.n_heads, self.context_window)
        self.stats = cp.zeros(stats_shape, dtype=cp.float32)

        output_shape = (
            self.batch_size,
            self.context_window,
            self.embedding_dim,
        )
        self.output = cp.zeros(output_shape, dtype=cp.float16)

        if "attention_input_grad" not in shared:
            # llmc has a 4 instead of a 3. it is unclear why. 3 seems to work fine for now
            input_grad_shape = (
                self.batch_size,
                self.context_window,
                3 * self.embedding_dim,
            )
            self.input_grad = cp.zeros(input_grad_shape, dtype=cp.float16)
            shared["attention_input_grad"] = self.input_grad
        else:
            self.input_grad = shared["attention_input_grad"]

        if "attention_output_grad" not in shared:
            output_grad_shape = (
                self.batch_size,
                self.context_window,
                self.embedding_dim,
            )
            self.output_grad = cp.zeros(output_grad_shape, dtype=cp.float16)
            shared["attention_output_grad"] = self.output_grad
        else:
            self.output_grad = shared["attention_output_grad"]

    def initialise_kernels(self):
        # Load the c functions
        lib = ctypes.CDLL(str(self.lib_path))
        self.stream = cp.cuda.get_current_stream()

        # initialise cuda/cublas/cudnn
        init = lib.initialize_cuda
        init.argtypes = [ctypes.c_void_p]
        init.restype = None
        init(ctypes.c_void_p(self.stream.ptr))

        # Define argument types for the kernels
        forward_args = [
            ctypes.c_void_p,  # float* out
            ctypes.c_void_p,  # float* stats
            ctypes.c_void_p,  # float* inp
            ctypes.c_int,  # int B
            ctypes.c_int,  # int T
            ctypes.c_int,  # int NH
            ctypes.c_int,  # int C
            ctypes.c_void_p,  # cudaStream_t stream
        ]
        self.forward_kernel = lib.attention_forward_cudnn
        self.forward_kernel.argtypes = forward_args
        self.forward_kernel.restype = None

        backward_args = [
            ctypes.c_void_p,  # float* dqkvr
            ctypes.c_void_p,  # float* dout
            ctypes.c_void_p,  # float* qkvr
            ctypes.c_void_p,  # float* o
            ctypes.c_void_p,  # float* stats
            ctypes.c_int,  # int B
            ctypes.c_int,  # int T
            ctypes.c_int,  # int NH
            ctypes.c_int,  # int C
            ctypes.c_void_p,  # cudaStream_t stream
        ]
        self.backward_kernel = lib.attention_backward_cudnn
        self.backward_kernel.argtypes = backward_args
        self.backward_kernel.restype = None

    def forward(self, tensor: Tensor):
        """
        Attention with a custom cuda kernel
        """
        if tensor.xp is not cp:
            raise ValueError("Cannot use numpy arrays with CUDNN")

        self.input = tensor

        self.forward_kernel(
            ctypes.c_void_p(self.output.data.ptr),  # float* out
            ctypes.c_void_p(self.stats.data.ptr),  # float* stats
            ctypes.c_void_p(self.input.array.data.ptr),  # float* inp
            ctypes.c_int(self.batch_size),  # int B
            ctypes.c_int(self.context_window),  # int T
            ctypes.c_int(self.n_heads),  # int NH
            ctypes.c_int(self.embedding_dim),  # int C
            ctypes.c_void_p(self.stream.ptr),
        )

        result = Tensor(self.output, is_batched=True)
        result.back_fns = (self.backward,)
        result.args = (self.input,)
        return result

    def backward(self, grad: Tensor):
        """
        Attention with a custom cuda kernel
        """
        if grad.xp is not cp:
            raise ValueError("Cannot use numpy arrays with CUDNN")
        self.output_grad = grad

        self.backward_kernel(
            ctypes.c_void_p(self.input_grad.data.ptr),  # float* dqkvr
            ctypes.c_void_p(self.output_grad.array.data.ptr),  # float* dout
            ctypes.c_void_p(self.input.array.data.ptr),  # float* qkvr
            ctypes.c_void_p(self.output.data.ptr),  # float* o
            ctypes.c_void_p(self.stats.data.ptr),  # float* stats
            ctypes.c_int(self.batch_size),  # int B
            ctypes.c_int(self.context_window),  # int T
            ctypes.c_int(self.n_heads),  # int NH
            ctypes.c_int(self.embedding_dim),  # int C
            ctypes.c_void_p(self.stream.ptr),  # cudaStream_t stream
        )

        return Tensor(self.input_grad)

    def to_gpu(self, *_):
        pass
