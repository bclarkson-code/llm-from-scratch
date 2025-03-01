import itertools
import math
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import torch
import triton
from matplotlib import pyplot as plt

from tricycle.activation import (
    CudaGeLU,
    CudaReLU,
    GeLU,
    ReLU,
    TritonGeLU,
    TritonRelu,
)
from tricycle.attention import Attention, TritonAttention
from tricycle.context import TRICYCLE_CONTEXT
from tricycle.kernels import (
    TritonAttentionRef,
    single_batched_matmul_kernel_1,
    single_batched_matmul_kernel_2,
)
from tricycle.layers import Dense, TritonDense
from tricycle.tensor import Tensor
from tricycle.utils import UseMixedPrecision


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_tokens"],
        # Modify x_vals to account for 3D shape
        x_vals=[
            2**i for i in range(8, 13)
        ],  # Reduced range since total elements will be cubed
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "tricycle"],
        line_names=["Triton", "Tricycle"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="attention-performance",
        args={},
    )
)
def benchmark(n_tokens, provider):
    # Create 3D input matrices

    quantiles = [0.5, 0.2, 0.8]
    batch_size, n_heads, n_tokens, head_size = (
        4,
        12,
        n_tokens,
        64,
    )

    causal = True
    tensor = np.random.normal(
        loc=0, scale=0.5, size=(batch_size, n_tokens, n_heads * head_size * 3)
    ).astype(np.float16)
    tensor = Tensor(tensor, dtype=np.float16, is_batched=True)
    tensor = tensor.to_gpu()

    match provider:
        case "triton":

            layer = TritonAttention(
                batch_size=batch_size,
                embedding_dim=n_heads * head_size,
                n_heads=n_heads,
                n_tokens=n_tokens,
            )

        case "tricycle":

            layer = Attention(
                embedding_dim=n_heads * head_size,
                n_heads=n_heads,
                context_window=n_tokens,
            )

        case _:
            raise ValueError(provider)
    layer.to_gpu()

    def fn():
        with UseMixedPrecision():
            layer(tensor)

        # layer.backward(y)

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(), quantiles=quantiles
    )

    # Update gbps calculation to account for 3D size
    def gbps(ms):
        flops_per_matmul = (
            2.0 * batch_size * n_heads * n_tokens * n_tokens * head_size
        )
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        return total_flops * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


def andrej_attention(
    q, k, v, B, T, C, n_head, block_size=32, sm_scale=0.5, bias=None
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
    # k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    # q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    # v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    att = q @ k.transpose(-2, -1)
    att *= sm_scale
    mask = torch.tril(torch.ones((T, T)).cuda())
    att[:, :, mask == 0] = float("-inf")
    att = F.softmax(att.to(torch.float32), dim=-1).half()
    y = att @ v
    return y
    # return y.transpose(1, 2).contiguous().view(B, T, C)


def compare_outputs(n_tokens, atol=1e-3, rtol=1e-3):
    """
    Compares the outputs of Triton and Tricycle attention implementations
    to ensure they produce the same results within specified tolerance.

    Args:
        n_tokens: Number of tokens for the test
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        bool: True if outputs match within tolerance, False otherwise
        dict: Dictionary with error statistics
    """
    # Setup parameters (same as in benchmark function)
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
    # tensor.requires_grad = True
    # q, k, v = tensor.split(head_size * n_heads, dim=-1)
    shape = (batch_size, n_heads, n_tokens, head_size)
    q = (
        torch.empty((shape), dtype=dtype)
        .cuda()
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((shape), dtype=dtype)
        .cuda()
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((shape), dtype=dtype)
        .cuda()
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    torch_out = andrej_attention(
        q,
        k,
        v,
        batch_size,
        n_tokens,
        head_size * n_heads,
        n_head=n_heads,
        block_size=n_tokens,
        sm_scale=sm_scale,
    )

    grad = torch.ones_like(torch_out)
    torch_out.backward(grad)

    # get triton output
    triton_ref = TritonAttentionRef()
    triton_output = triton_ref.forward(
        deepcopy(q), deepcopy(k), deepcopy(v), True, sm_scale
    )

    assert torch.allclose(triton_output, torch_out, rtol=1e-3, atol=1e-3)
    triton_grad, dq, dk, dv = triton_ref.backward(grad)

    q_diff = abs(q.grad - dq)
    k_diff = abs(k.grad - dk)
    v_diff = abs(v.grad - dv)

    breakpoint()

    # why do q and v match but not k?
    assert torch.allclose(v.grad, dv, atol=1e-2, rtol=0)  # works
    assert torch.allclose(q.grad, dq, atol=1e-2, rtol=0)  # works
    assert torch.allclose(k.grad, dk, atol=1e-2, rtol=0)  # works

    # assert torch.allclose(triton_grad, tensor.grad)
    raise Exception("Done")

    # Get Tricycle output
    tricycle_layer = TritonAttention(
        batch_size=batch_size,
        n_tokens=n_tokens,
        embedding_dim=n_heads * head_size,
        n_heads=n_heads,
    )
    tricycle_layer.sm_scale = sm_scale
    tricycle_layer.to_gpu()
    TRICYCLE_CONTEXT.use_mixed_precision = True
    tricycle_output_raw = tricycle_layer.forward(tricycle_tensor)

    # Convert Tricycle output to PyTorch tensor for comparison
    # Assuming Tricycle output can be converted this way - adjust if needed
    tricycle_output = torch.tensor(
        tricycle_output_raw.cpu().numpy(), dtype=dtype, device=DEVICE
    )

    # Check shapes first
    shape_match = torch_out.shape == tricycle_output.shape

    if not shape_match:
        return False, {
            "shape_match": False,
            "triton_shape": torch_out.shape,
            "tricycle_shape": tricycle_output.shape,
        }

    # Calculate differences
    abs_diff = torch.abs(torch_out - tricycle_output)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()

    # Relative differences (avoiding division by zero)
    eps = 1e-8
    rel_diff = abs_diff / (torch.abs(torch_out) + eps)
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()

    # Check if within tolerance
    is_close = torch.allclose(torch_out, tricycle_output, atol=atol, rtol=rtol)

    # Return results
    stats = {
        "shape_match": shape_match,
        "max_absolute_diff": max_abs_diff,
        "mean_absolute_diff": mean_abs_diff,
        "max_relative_diff": max_rel_diff,
        "mean_relative_diff": mean_rel_diff,
        "within_tolerance": is_close,
        "atol_used": atol,
        "rtol_used": rtol,
    }

    return is_close, stats


def test_all_sizes(atol=1e-3, rtol=1e-3):
    """
    Tests output comparison for all sizes used in the benchmark.

    Args:
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        dict: Results for each tested size
    """
    # Same token sizes as in the benchmark

    results = {}
    for n_tokens in [256, 512, 1024]:

        print(f"Testing with n_tokens = {n_tokens}")
        is_match, stats = compare_outputs(n_tokens, atol, rtol)
        results[n_tokens] = {"match": is_match, "stats": stats}

        if not is_match:
            print(f"❌ Outputs do not match for n_tokens = {n_tokens}")
            print(f"Max absolute difference: {stats['max_absolute_diff']}")
            print(f"Max relative difference: {stats['max_relative_diff']}")
        else:
            print(
                f"✓ Outputs match within tolerance for n_tokens = {n_tokens}"
            )

    return results


if __name__ == "__main__":
    # Test with default tolerances
    results = test_all_sizes()

    # Print summary
    all_match = all(result["match"] for result in results.values())

    if all_match:
        print("\n✓ All tests passed! Outputs match within tolerance.")
    else:
        print("\n❌ Some tests failed. Check individual results.")

    # benchmark.run(print_data=True, show_plots=True)
    # plt.savefig("fig.png")
