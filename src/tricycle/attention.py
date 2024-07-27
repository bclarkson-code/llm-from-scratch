"""Attention module for multi-head attention operations.

This module implements the multi-head attention mechanism as described in
"Attention Is All You Need" (Vaswani et al., 2017). It includes functions for
building attention masks and the main Attention class for performing
multi-head attention operations.
"""

from math import sqrt

import cudnn
import numpy as np

from tricycle import GPU_ENABLED
from tricycle.context import TRICYCLE_CONTEXT
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
                (self.batch_size, self.context_window, self.embedding_dim * 3)
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

    def to_gpu(self, device: int):
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
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.context_window = context_window

        # this is where we'll store intemediate results
        self.stats = None
        self.output = None

        self.graph_forward = None
        self.graph_backward = None

        # used by flash attention for temporary storage
        self._workspace = None

    def backward(self, grad: Tensor):
        xp = grad.xp

        B, T, _ = grad.array.shape
        NH = self.n_heads
        C = self.embedding_dim
        HS = C // NH

        result = xp.empty((B, T, 3, NH, HS), dtype=xp.float16)

        query = self.qkv[:, :, 0].reshape(B, NH, T, HS)
        key = self.qkv[:, :, 1].reshape(B, NH, T, HS)
        value = self.qkv[:, :, 2].reshape(B, NH, T, HS)

        query_result = result[:, :, 0].reshape(B, NH, T, HS)
        key_result = result[:, :, 1].reshape(B, NH, T, HS)
        value_result = result[:, :, 2].reshape(B, NH, T, HS)

        if self.graph_backward is None:
            self.graph_backward = cudnn.pygraph(
                io_data_type=cudnn.data_type.HALF,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            q_stride = (NH * HS * T, HS, NH * HS, 1)
            k_stride = (NH * HS * T, HS, NH * HS, 1)
            v_stride = (NH * HS * T, HS, NH * HS, 1)
            grad_stride = (NH * HS * T, HS, NH * HS, 1)
            o_stride = (NH * HS * T, HS, NH * HS, 1)
            stats_stride = (NH * T, T, 1, 1)

            self.q_backward = self.graph_backward.tensor(
                dim=(B, NH, T, HS),
                stride=q_stride,
                data_type=cudnn.data_type.HALF,
            )
            self.k_backward = self.graph_backward.tensor(
                dim=(B, NH, T, HS),
                stride=k_stride,
                data_type=cudnn.data_type.HALF,
            )
            self.v_backward = self.graph_backward.tensor(
                dim=(B, NH, T, HS),
                stride=v_stride,
                data_type=cudnn.data_type.HALF,
            )
            self.grad_backward = self.graph_backward.tensor(
                dim=(B, NH, T, HS),
                stride=grad_stride,
                data_type=cudnn.data_type.HALF,
            )
            self.o_backward = self.graph_backward.tensor(
                dim=(B, NH, T, HS),
                stride=o_stride,
                data_type=cudnn.data_type.HALF,
            )
            self.stats_backward = self.graph_backward.tensor(
                dim=(B, NH, T, 1),
                stride=stats_stride,
                data_type=cudnn.data_type.FLOAT,
            )
            # training mode in enabled with is_inference=False
            # causal mask is enabled
            (
                self.q_grad_backward,
                self.k_grad_backward,
                self.v_grad_backward,
            ) = self.graph_backward.sdpa_backward(
                name="sdpa_backward",
                q=self.q_backward,
                k=self.k_backward,
                v=self.v_backward,
                o=self.o_backward,
                dO=self.grad_backward,
                stats=self.stats_backward,
                attn_scale=self.attn_scale,
                use_causal_mask=True,
            )
            dim = (B, NH, T, HS)
            self.q_grad_backward.set_output(True).set_dim(dim).set_stride(
                q_stride
            )
            self.k_grad_backward.set_output(True).set_dim(dim).set_stride(
                k_stride
            )
            self.v_grad_backward.set_output(True).set_dim(dim).set_stride(
                v_stride
            )

            self.graph_backward.validate()
            self.graph_backward.build_operation_graph()
            self.graph_backward.create_execution_plans(
                [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
            )
            self.graph_backward.check_support()
            self.graph_backward.build_plans()
            workspace_size = max(
                self.graph_forward.get_workspace_size(),
                self.graph_backward.get_workspace_size(),
            )
            self._workspace = xp.empty(workspace_size * 2, dtype=xp.uint8)

        variant_pack_backward = {
            self.q_backward: query,
            self.k_backward: key,
            self.v_backward: value,
            self.o_backward: self.output,
            self.grad_backward: grad.array,
            self.stats_backward: self.stats,
            self.q_grad_backward: query_result,
            self.k_grad_backward: key_result,
            self.v_grad_backward: value_result,
        }
        for tensor in [query, key, value, self.output, grad.array, self.stats]:
            assert (
                tensor.data.ptr % 16 == 0
            ), f"Tensor {tensor} is not 16-byte aligned"
        self.graph_backward.execute(variant_pack_backward, self._workspace)
        xp.cuda.stream.get_current_stream().synchronize()
        xp.cuda.runtime.deviceSynchronize()
        breakpoint()

        return Tensor(result)

    def forward(self, tensor: Tensor):
        xp = tensor.xp

        assert tensor.is_batched

        B, T, _ = tensor.array.shape
        NH = self.n_heads
        C = self.embedding_dim
        HS = C // NH

        # tensor is a qkv tensor
        self.qkv = tensor.array.reshape(B, T, 3, NH, HS)
        query = self.qkv[:, :, 0].reshape(B, NH, T, HS)
        key = self.qkv[:, :, 1].reshape(B, NH, T, HS)
        value = self.qkv[:, :, 2].reshape(B, NH, T, HS)

        self.attn_scale = 1.0 / sqrt(self.n_heads)
        if self.stats is None:
            self.stats = xp.empty((B, T, C), dtype=xp.float32)

        if self.output is None:
            self.output = xp.empty((B, T, C), dtype=query.dtype)

        # if it is the first time, compile a computational graph with cudnn
        if self.graph_forward is None:
            self.graph_forward = cudnn.pygraph(
                io_data_type=cudnn.data_type.HALF,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            q_stride = (NH * HS * T, HS, NH * HS, 1)
            k_stride = (NH * HS * T, HS, NH * HS, 1)
            v_stride = (NH * HS * T, HS, NH * HS, 1)

            self.q_forward = self.graph_forward.tensor(
                dim=(B, NH, T, HS),
                stride=q_stride,
                data_type=cudnn.data_type.HALF,
            )
            self.k_forward = self.graph_forward.tensor(
                dim=(B, NH, T, HS),
                stride=k_stride,
                data_type=cudnn.data_type.HALF,
            )
            self.v_forward = self.graph_forward.tensor(
                dim=(B, NH, T, HS),
                stride=v_stride,
                data_type=cudnn.data_type.HALF,
            )
            # training mode in enabled with is_inference=False
            # causal mask is enabled
            self.o_forward, self.stats_forward = self.graph_forward.sdpa(
                name="sdpa",
                q=self.q_forward,
                k=self.k_forward,
                v=self.v_forward,
                is_inference=False,
                attn_scale=self.attn_scale,
                use_causal_mask=True,
            )

            o_stride = (NH * HS * T, HS, NH * HS, 1)
            self.o_forward.set_output(True).set_dim([B, NH, T, HS]).set_stride(
                o_stride
            )
            self.stats_forward.set_output(True).set_dim(
                [B, NH, T, 1]
            ).set_stride([NH * T, T, 1, 1]).set_data_type(
                cudnn.data_type.FLOAT
            )

            self.graph_forward.validate()
            self.graph_forward.build_operation_graph()
            self.graph_forward.create_execution_plans(
                [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
            )
            self.graph_forward.check_support()
            self.graph_forward.build_plans()
            workspace_size = self.graph_forward.get_workspace_size()
            if (
                self._workspace is None
                or self._workspace.size() < workspace_size
            ):
                self._workspace = xp.empty(workspace_size, dtype=xp.uint8)

        variant_pack_forward = {
            self.q_forward: query,
            self.k_forward: key,
            self.v_forward: value,
            self.o_forward: self.output,
            self.stats_forward: self.stats,
        }
        self.graph_forward.execute(variant_pack_forward, self._workspace)
        xp.cuda.stream.get_current_stream().synchronize()

        return Tensor(
            self.output,
            is_batched=True,
            back_fns=(self.backward,),
            args=(tensor,),
        )

    def to_gpu(self, device: int = 0):
        pass

    def from_gpu(self):
        pass
