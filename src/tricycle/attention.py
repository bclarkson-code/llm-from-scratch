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

        self.forward_graph = None
        self.graph_backward = None

        # used by flash attention for temporary storage
        self._workspace = None

    def _build_forward_graph(self, query, key, value, output):
        graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.HALF,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        B, NH, T, HS = query.shape
        dim = (B, NH, T, HS)
        strides = (NH * T * HS, T * HS, HS, 1)  # BNTH layout
        q = graph.tensor(
            dim=dim, stride=strides, data_type=cudnn.data_type.HALF
        )
        k = graph.tensor(
            dim=dim, stride=strides, data_type=cudnn.data_type.HALF
        )
        v = graph.tensor(
            dim=dim, stride=strides, data_type=cudnn.data_type.HALF
        )
        o = graph.tensor(
            dim=dim, stride=strides, data_type=cudnn.data_type.HALF
        )
        o, stats = graph.sdpa(
            name="sdpa",
            q=q,
            k=k,
            v=v,
            is_inference=False,
            attn_scale=self.attn_scale,
            use_causal_mask=True,
        )

        o.set_output(True).set_dim(dim).set_stride(strides).set_data_type(
            cudnn.data_type.HALF
        )
        stats.set_output(True).set_dim((B, NH, T, 1))

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans(
            [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
        )
        graph.check_support()
        graph.build_plans()

        tensors = {"q": q, "k": k, "v": v, "o": o, "stats": stats}
        return graph, tensors

    def _build_backward_graph(self):
        graph = cudnn.create_graph()

        B, NH, T, HS = query.shape
        q = graph.create_tensor((B, NH, T, HS))
        k = graph.create_tensor((B, NH, T, HS))
        v = graph.create_tensor((B, NH, T, HS))
        o = graph.create_tensor((B, NH, T, HS))
        do = graph.create_tensor((B, NH, T, HS))
        stats = graph.create_tensor((B, NH, T, 1))

        dq, dk, dv = graph.sdpa_backward(
            name="sdpa_backward",
            q=q,
            k=k,
            v=v,
            o=o,
            dO=do,
            stats=stats,
            attn_scale=self.attn_scale,
            use_causal_mask=True,
        )

        dq.set_output(True)
        dk.set_output(True)
        dv.set_output(True)

        graph.compile()

        tensors = {
            "q": q,
            "k": k,
            "v": v,
            "o": o,
            "do": do,
            "stats": stats,
            "dq": dq,
            "dk": dk,
            "dv": dv,
        }

        return graph, tensors

    def forward(self, tensor: Tensor):
        import cupy as cp

        batch_size = 2  # batch size
        n_heads = 4  # query number of heads
        embedding_dim = 10  # maximum sequence length
        head_size = 64  # embedding dimension per head

        self.attn_scale = 1 / sqrt(head_size)
        strides = (
            embedding_dim * n_heads * head_size,
            head_size,
            n_heads * head_size,
            1,
        )
        # Reshape qkv to BS3HD layout
        self.qkv = tensor.array.reshape(
            (batch_size, embedding_dim, 3, n_heads, head_size)
        )

        # Transpose to 3BNTHD layout for easier slicing
        storage = self.qkv.transpose(2, 0, 1, 3, 4)

        # Create separate views for Q, K, V
        self.query = storage[0]
        self.key = storage[1]
        self.value = storage[2]

        # Reshape to BNTHD layout as expected by cuDNN
        self.query = self.query.reshape(
            batch_size, n_heads, embedding_dim, head_size
        )
        self.key = self.key.reshape(
            batch_size, n_heads, embedding_dim, head_size
        )
        self.value = self.value.reshape(
            batch_size, n_heads, embedding_dim, head_size
        )

        self.output = cp.empty(
            (batch_size, n_heads, embedding_dim, head_size),
            dtype=self.qkv.dtype,
            order="C",
        )
        self.stats = cp.empty(
            (batch_size, n_heads, embedding_dim, 1),
            dtype=cp.float32,
            order="C",
        )

        if self.forward_graph is None:
            self.forward_graph, self.forward_tensors = (
                self._build_forward_graph(
                    self.query, self.key, self.value, self.output
                )
            )

        workspace_size = self.forward_graph.get_workspace_size()
        workspace = cp.empty((workspace_size,), dtype=cp.uint8)

        self.forward_graph.execute(
            {
                self.forward_tensors["q"]: self.query,
                self.forward_tensors["k"]: self.key,
                self.forward_tensors["v"]: self.value,
                self.forward_tensors["o"]: self.output,
                self.forward_tensors["stats"]: self.stats,
            },
            workspace,
        )

        # Add debugging output
        print(f"Output shape: {self.output.shape}")
        print(f"Output dtype: {self.output.dtype}")
        print(
            f"Output memory info: {self.output.data.mem.ptr}, {self.output.data.mem.size}"
        )

        # Reshape o to match input shape
        try:
            self.output = self.output.transpose(0, 2, 1, 3).reshape(
                batch_size, embedding_dim, n_heads * head_size
            )
        except Exception as e:
            print(f"Error during output reshaping: {e}")
            raise
        return Tensor(
            self.output,
            back_fns=(self.backward,),
            args=(self.qkv,),
            is_batched=True,
            dtype=cp.float16,
        )

    def backward(self, dout, q, k, v, o, stats):
        import cupy as cp

        B, T, NH_HS = dout.shape
        assert NH_HS == NH * HS, "dout shape mismatch"

        # Reshape dout to match SDPA output shape
        do = dout.reshape(B, T, NH, HS).transpose(0, 2, 1, 3)

        dq = cp.empty_like(q)
        dk = cp.empty_like(k)
        dv = cp.empty_like(v)

        workspace_size = self.backward_graph.get_workspace_size()
        workspace = cp.empty((workspace_size,), dtype=cp.uint8)

        self.backward_graph.execute(
            {
                self.backward_tensors["q"]: q,
                self.backward_tensors["k"]: k,
                self.backward_tensors["v"]: v,
                self.backward_tensors["o"]: o,
                self.backward_tensors["do"]: do,
                self.backward_tensors["stats"]: stats,
                self.backward_tensors["dq"]: dq,
                self.backward_tensors["dk"]: dk,
                self.backward_tensors["dv"]: dv,
            },
            workspace,
        )

        # Combine dq, dk, dv into a single array
        dqkv = cp.stack([dq, dk, dv], axis=2)
        dqkv = dqkv.transpose(0, 2, 3, 1, 4).reshape(B, T, 3 * NH * HS)

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
