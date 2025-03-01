import math
from copy import deepcopy

import numpy as np
import pytest
import torch
import triton
from matplotlib import pyplot as plt

from tricycle import GPU_ENABLED
from tricycle.attention import Attention, TritonAttention, build_mask
from tricycle.blocks import MultiHeadSelfAttention
from tricycle.context import TRICYCLE_CONTEXT
from tricycle.einsum import Einsum
from tricycle.exceptions import GPUDisabledException
from tricycle.functions import Softmax
from tricycle.kernels import (
    TritonAttentionRef,
    single_batched_matmul_kernel_1,
    single_batched_matmul_kernel_2,
)
from tricycle.tensor import DEFAULT_DTYPE, Tensor
from tricycle.utils import UseMixedPrecision

TORCH_DTYPE = (
    torch.float16 if TRICYCLE_CONTEXT.use_mixed_precision else torch.float32
)


def pytorch_attention(q, k, v, B, T, C, n_head):
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    # return k
    y = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0,
        is_causal=True,
    )
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return y


def andrej_attention(
    q, k, v, B, T, C, n_head, block_size=32, bias=None, dtype=TORCH_DTYPE
):
    """
    Andrej Karpathy's implementation of attention from nanogpt
    """
    import math

    from torch.nn import functional as F

    if bias is None:
        bias = (
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
            .to(q.device)
        )
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    att = (q.to(torch.float32) @ k.to(torch.float32).transpose(-2, -1)).to(
        TORCH_DTYPE
    )
    att *= 1.0 / math.sqrt(k.size(-1))
    att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att.to(torch.float32), dim=-1)
    y = att @ v.to(
        torch.float32
    )  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    return y.to(dtype).transpose(1, 2).contiguous().view(B, T, C)


def andrej_attention_block(
    x, B, T, C, n_head, c_attn, c_proj, n_embd, block_size=32
):
    """
    Andrej Karpathy's implementation of an attention block from nanogpt
    """
    q, k, v = c_attn(x).split(n_embd, dim=-1)
    y = andrej_attention(q, k, v, B, T, C, n_head, block_size)
    return c_proj(y.to(torch.float32)).to(TORCH_DTYPE)


def test_attention_individually():
    """
    This operation is pretty complex so we'll perform each stage
    with pytorch and then compare the results. Here, I'm comparing
    with Andrej Karpathy's implementation from NanoGPT
    For this test, we're doing everything non-batch
    """
    # setup
    embedding_dim = 15
    n_heads = 3
    n_tokens = 7
    projected_size = embedding_dim * 3
    context_window = 32
    head_size = embedding_dim // n_heads
    head_shape = (n_tokens, n_heads, head_size)
    out_shape = (n_tokens, embedding_dim)
    T = n_tokens
    C = embedding_dim

    # random input tensor
    in_tensor = np.random.uniform(-5, 5, (n_tokens, projected_size))
    in_tensor = Tensor(in_tensor)
    xp = in_tensor.xp

    x = torch.from_numpy(in_tensor.array)

    qu, k, v = x.split(embedding_dim, dim=-1)  # pytorch
    query, key, value = in_tensor.split(3, axis=-1)  # tricycle

    assert query.close_to(qu, rtol=1e-3)
    assert key.close_to(k, rtol=1e-3)
    assert value.close_to(v, rtol=1e-3)

    # pytorch
    k = k.view(T, n_heads, C // n_heads)
    qu = qu.view(T, n_heads, C // n_heads)
    v = v.view(T, n_heads, C // n_heads)
    k = k.transpose(-3, -2)
    qu = qu.transpose(-3, -2)
    v = v.transpose(-3, -2)

    # tricycle
    key = key.reshape(head_shape).einsum("TNH -> NTH")
    query = query.reshape(head_shape).einsum("TNH -> NTH")
    value = value.reshape(head_shape).einsum("TNH -> NTH")

    assert query.close_to(qu, rtol=1e-3)
    assert key.close_to(k, rtol=1e-3)
    assert value.close_to(v, rtol=1e-3)

    # pytorch
    att = qu.to(torch.float32) @ k.transpose(-2, -1).to(torch.float32)
    att = att.to(TORCH_DTYPE)
    att *= 1 / np.sqrt(k.size(-1))

    # tricycle
    attention = Einsum("NIh, NJh -> NIJ")(query, key) / np.sqrt(head_size)

    assert attention.close_to(att, rtol=1e-1, atol=1e-4)

    # pytorch
    bias = torch.tril(torch.ones(context_window, context_window)).view(
        1, context_window, context_window
    )
    att = att.masked_fill(bias[:, :T, :T] == 0, float("-inf"))

    # tricycle
    mask = build_mask(context_window, n_heads=n_heads)
    attention = xp.where(
        mask[:, :n_tokens, :n_tokens], -xp.inf, attention.array
    )
    attention = Tensor(attention)

    assert attention.close_to(att.numpy(), rtol=1e-2)

    # pytorch
    att = torch.softmax(att.to(torch.float32), dim=-1)

    # tricycle
    attention = Softmax()(attention)

    assert attention.close_to(att.numpy(), rtol=1e-2, atol=1e-4)

    # pytorch
    att = att.to(torch.float32) @ v.to(torch.float32)
    att = att.to(TORCH_DTYPE).transpose(0, 1).contiguous()

    # tricycle
    attention = Einsum("NIj, NjH -> INH")(attention, value)

    assert attention.close_to(att.numpy(), rtol=1e-2)

    # pytorch
    att = att.view(T, C)

    # tricycle
    attention = attention.reshape(out_shape)

    assert attention.close_to(att.numpy(), rtol=1e-2)


def test_attention_combined():
    """
    Compare Tricycle's attention with Andrej's
    """
    n_heads = 3
    embedding_dim = 15
    n_tokens = 7
    batch_size = 11
    projected_size = embedding_dim * 3
    context_window = n_tokens
    B = batch_size
    T = n_tokens
    C = embedding_dim

    np.random.seed(0)
    # random input tensor
    in_tensor = np.random.uniform(
        -5, 5, (batch_size, n_tokens, projected_size)
    )
    in_tensor = Tensor(in_tensor).to_batched()

    x = torch.from_numpy(in_tensor.array)
    x.requires_grad = True

    qu, k, v = x.split(embedding_dim, dim=-1)

    pytorch_result = andrej_attention(
        q=qu,
        k=k,
        v=v,
        B=B,
        T=T,
        C=C,
        n_head=n_heads,
    )

    tricycle_attention = Attention(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
    )
    tricycle_result = tricycle_attention(in_tensor).from_batched()

    assert tricycle_result.close_to(
        pytorch_result.detach(), rtol=1e-1, atol=1e-3
    )

    tricycle_result.from_batched().sum().backward()
    pytorch_result.sum().backward()

    # I don't know why the tolerance has to be so large here.
    # smells like numerical instability
    # TODO: investigate discrepency
    assert in_tensor.grad.close_to(
        x.grad.detach().numpy(), atol=1e-1, rtol=1e-1
    )


def test_attention_block():
    """
    Compare Tricycle attention with pytorch's MultiheadAttention
    """
    n_heads = 3
    embedding_dim = 15
    n_tokens = 32
    batch_size = 11
    context_window = 32

    np.random.seed(0)

    x = np.random.normal(size=(batch_size, n_tokens, embedding_dim)).astype(
        DEFAULT_DTYPE
    )

    in_projection_weights = np.random.normal(
        0, 1, (embedding_dim, embedding_dim * 3)
    ).astype(DEFAULT_DTYPE)
    out_projection_weights = np.random.normal(
        0, 1, (embedding_dim, embedding_dim)
    ).astype(DEFAULT_DTYPE)

    tricycle_attention = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        residual_dropout_prob=0,
    )
    tricycle_attention.in_projection.weights = Tensor(
        in_projection_weights, name="in_proj"
    )
    tricycle_attention.out_projection.weights = Tensor(
        out_projection_weights, name="out_proj"
    )

    in_tensor = Tensor(x, requires_grad=False).to_batched()
    tricycle_result = tricycle_attention(in_tensor)

    c_attn = torch.nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
    c_attn.weight = torch.nn.Parameter(torch.tensor(in_projection_weights.T))
    c_proj = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
    c_proj.weight = torch.nn.Parameter(torch.tensor(out_projection_weights.T))

    andrej_result = andrej_attention_block(
        torch.tensor(x),
        batch_size,
        n_tokens,
        embedding_dim,
        n_heads,
        c_attn,
        c_proj,
        embedding_dim,
        block_size=32,
    )

    assert tricycle_result.close_to(
        andrej_result.detach().numpy(), rtol=1e-1, atol=1e-1
    )

    tricycle_loss = tricycle_result.from_batched().einsum("abc->")
    andrej_loss = andrej_result.sum()

    assert tricycle_loss.close_to(andrej_loss.detach().numpy())

    tricycle_loss.backward()
    andrej_loss.backward()

    assert not tricycle_attention.out_projection.weights.is_batched
    tricycle_out_weights = tricycle_attention.out_projection.weights.grad

    assert tricycle_out_weights.close_to(c_proj.weight.grad.T.numpy())

    tricycle_in_weights = tricycle_attention.in_projection.weights.grad

    assert tricycle_in_weights.close_to(
        c_attn.weight.grad.T.numpy(), rtol=1e-2, atol=1e-4
    )


def generate_causal_mask(seq_length):
    import torch

    # Create a causal mask
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    return mask


# test_cudnn_attention_vs_pytorch has been removed


def test_triton_attention_correctness_fp16():
    """
    Tests that TritonAttention produces results consistent with PyTorch and TritonAttentionRef
    using float16 precision.
    """
    if not GPU_ENABLED:
        pytest.skip(
            "Cannot run test_triton_attention_correctness_fp16 if GPU is not available"
        )

    # Setup parameters
    n_tokens = 256
    batch_size, n_heads, head_size = 4, 12, 64

    DEVICE = torch.device("cuda:0")
    dtype = torch.float16

    torch.manual_seed(0)

    # Create input tensor
    tensor = torch.empty(
        (batch_size, n_tokens, n_heads * head_size * 3),
        dtype=dtype,
        device=DEVICE,
    ).normal_(mean=0.0, std=0.5)
    tricycle_tensor = Tensor(
        deepcopy(tensor).cpu(), is_batched=True, dtype=np.float16
    ).to_gpu()
    sm_scale = 1 / math.sqrt(head_size)

    # Get torch output
    q, k, v = tensor.split(head_size * n_heads, dim=-1)
    torch_out = andrej_attention(
        q,
        k,
        v,
        batch_size,
        n_tokens,
        head_size * n_heads,
        n_head=n_heads,
        block_size=n_tokens,
        dtype=torch.float16,
    )

    # Get triton reference output
    triton_output = TritonAttentionRef().forward(tensor, True, sm_scale)

    # Verify TritonAttentionRef matches PyTorch with float16
    assert torch.allclose(
        triton_output, torch_out, rtol=1e-2, atol=1e-2
    ), "TritonAttentionRef and PyTorch outputs don't match in float16"

    # Get Tricycle output
    with UseMixedPrecision():
        tricycle_layer = TritonAttention(
            batch_size=batch_size,
            n_tokens=n_tokens,
            embedding_dim=n_heads * head_size,
            n_heads=n_heads,
        )
        tricycle_layer.sm_scale = sm_scale
        tricycle_layer.to_gpu()
        tricycle_output_raw = tricycle_layer.forward(tricycle_tensor)

    # Convert Tricycle output to PyTorch tensor for comparison
    tricycle_output = torch.tensor(
        tricycle_output_raw.cpu().numpy(), dtype=dtype, device=DEVICE
    )

    # Verify shapes match
    assert (
        torch_out.shape == tricycle_output.shape
    ), "Output shapes don't match"

    # Verify values are close using float16 precision
    assert torch.allclose(
        torch_out, tricycle_output, atol=1e-2, rtol=1e-2
    ), "Outputs from TritonAttention don't match reference implementation in float16"


def test_triton_attention_correctness_fp32():
    """
    Tests that TritonAttention produces results consistent with PyTorch and TritonAttentionRef
    using float32 precision.
    """
    if not GPU_ENABLED:
        pytest.skip(
            "Cannot run test_triton_attention_correctness_fp32 if GPU is not available"
        )

    # Setup parameters
    n_tokens = 256
    batch_size, n_heads, head_size = 4, 12, 64

    DEVICE = torch.device("cuda:0")

    # Use float16 for input but convert to float32 for comparison
    dtype_input = torch.float16
    dtype_compare = torch.float32

    torch.manual_seed(0)

    # Create input tensor
    tensor = torch.empty(
        (batch_size, n_tokens, n_heads * head_size * 3),
        dtype=dtype_input,
        device=DEVICE,
    ).normal_(mean=0.0, std=0.5)
    tricycle_tensor = Tensor(
        deepcopy(tensor).cpu(), is_batched=True, dtype=np.float16
    ).to_gpu()
    sm_scale = 1 / math.sqrt(head_size)

    # Get torch output
    q, k, v = tensor.split(head_size * n_heads, dim=-1)
    torch_out = andrej_attention(
        q,
        k,
        v,
        batch_size,
        n_tokens,
        head_size * n_heads,
        n_head=n_heads,
        block_size=n_tokens,
    )

    # Get triton reference output
    triton_output = TritonAttentionRef().forward(tensor, True, sm_scale)

    # Verify TritonAttentionRef matches PyTorch with float32 conversion
    assert torch.allclose(
        triton_output.to(dtype_compare),
        torch_out.to(dtype_compare),
        rtol=1e-3,
        atol=1e-3,
    ), "TritonAttentionRef and PyTorch outputs don't match in float32"

    # Get Tricycle output
    with UseMixedPrecision():
        tricycle_layer = TritonAttention(
            batch_size=batch_size,
            n_tokens=n_tokens,
            embedding_dim=n_heads * head_size,
            n_heads=n_heads,
        )
        tricycle_layer.sm_scale = sm_scale
        tricycle_layer.to_gpu()
        tricycle_output_raw = tricycle_layer.forward(tricycle_tensor)

    # Convert Tricycle output to PyTorch tensor for comparison
    tricycle_output = torch.tensor(
        tricycle_output_raw.cpu().numpy(), dtype=dtype_input, device=DEVICE
    )

    # Verify shapes match
    assert (
        torch_out.shape == tricycle_output.shape
    ), "Output shapes don't match"

    # Verify values are close using float32 precision
    assert torch.allclose(
        torch_out.to(dtype_compare),
        tricycle_output.to(dtype_compare),
        atol=1e-3,
        rtol=1e-3,
    ), "Outputs from TritonAttention don't match reference implementation in float32"


if __name__ == "__main__":
    try:
        print("Testing with float16 precision...")
        test_triton_attention_correctness_fp16()
        print("✅ Float16 test passed")
    except (AssertionError, RuntimeError) as e:
        print(f"❌ Float16 test failed: {e}")

    try:
        print("\nTesting with float32 precision...")
        test_triton_attention_correctness_fp32()
        print("✅ Float32 test passed")
    except (AssertionError, RuntimeError) as e:
        print(f"❌ Float32 test failed: {e}")
