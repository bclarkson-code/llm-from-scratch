import numpy as np

from tricycle.activation import CudaGeLU, CudaReLU, GeLU, ReLU, TritonRelu
from tricycle.attention import Attention, CudnnAttention
from tricycle.layers import CudaDense, Dense
from tricycle.tensor import Tensor
from tricycle.utils import UseMixedPrecision

N_LOOPS = 100
# INPUT_SHAPE = (2, 64, 16)
OUTPUT_SHAPE = 8
N_HEADS = 2

INPUT_SHAPE = (16, 1024, 768)
# OUTPUT_SHAPE = 768
# N_HEADS = 12


def bench_vanilla_relu():
    np.random.seed(0)
    tensor = Tensor((np.random.random(INPUT_SHAPE) * 2 - 1))
    tensor.to_gpu()
    layer = ReLU()
    layer.to_gpu()
    for _ in range(N_LOOPS):
        output = layer(tensor)
        # output.backward()


def bench_cuda_relu():
    np.random.seed(0)
    tensor = Tensor((np.random.random(INPUT_SHAPE) * 2 - 1))
    tensor.to_gpu()
    layer = CudaReLU()
    layer.to_gpu()
    for _ in range(N_LOOPS):
        output = layer(tensor)
        # output.backward()


def bench_triton_relu():
    np.random.seed(0)
    tensor = Tensor((np.random.random(INPUT_SHAPE) * 2 - 1))
    tensor.to_gpu()
    layer = TritonRelu()
    for _ in range(N_LOOPS):
        output = layer(tensor)
        # output.backward()


def bench_vanilla_dense():
    np.random.seed(0)
    tensor = Tensor(np.random.random(INPUT_SHAPE) * 2 - 1)
    tensor.to_gpu()
    layer = Dense(INPUT_SHAPE[-1], INPUT_SHAPE[-1])
    layer.to_gpu()
    for _ in range(N_LOOPS):
        output = layer(tensor)
        output.backward()


def bench_cuda_dense():
    np.random.seed(0)
    tensor = Tensor((np.random.random(INPUT_SHAPE) * 2 - 1))
    tensor.to_gpu()
    layer = CudaDense(INPUT_SHAPE[-1], INPUT_SHAPE[-1])
    layer.to_gpu()
    for _ in range(N_LOOPS):
        output = layer(tensor)
        output.backward()


def bench_vanilla_gelu():
    np.random.seed(0)
    tensor = Tensor((np.random.random(INPUT_SHAPE) * 2 - 1))
    tensor.to_gpu()
    layer = GeLU()
    for _ in range(N_LOOPS):
        output = layer(tensor)
        output.backward()


def bench_cuda_gelu():
    np.random.seed(0)
    tensor = Tensor((np.random.random(INPUT_SHAPE) * 2 - 1))
    tensor.to_gpu()
    layer = CudaGeLU()
    for _ in range(N_LOOPS):
        output = layer(tensor)
        output.backward()


def bench_vanilla_attention():
    with UseMixedPrecision():
        np.random.seed(0)
        B, T, C = INPUT_SHAPE
        shape = B, T, C * 3
        tensor = Tensor(
            (np.random.random(shape).astype(np.float16) * 2 - 1),
            is_batched=True,
        )
        layer = Attention(embedding_dim=C, n_heads=N_HEADS, context_window=T)
        tensor.to_gpu()
        layer.to_gpu()
        for _ in range(N_LOOPS):
            output = layer(tensor)
            output.backward()


def bench_cudnn_attention():
    with UseMixedPrecision():
        np.random.seed(0)
        B, T, C = INPUT_SHAPE
        shape = B, T, C * 3
        tensor = Tensor(
            (np.random.random(shape).astype(np.float16) * 2 - 1),
            is_batched=True,
        )
        layer = CudnnAttention(
            batch_size=B,
            embedding_dim=C,
            n_heads=N_HEADS,
            context_window=T,
            shared={},
        )
        tensor.to_gpu()
        # layer.to_gpu()
        for _ in range(N_LOOPS):
            output = layer(tensor)
            output.backward()


def test_attention_match():
    np.random.seed(0)
    import cupy as cp

    B, T, C = INPUT_SHAPE
    shape = B, T, C * 3
    random_data = np.random.random(shape).astype(np.float32) * 2 - 1
    tensor_1 = Tensor(random_data.copy(), is_batched=True)
    tensor_2 = Tensor(random_data.copy(), is_batched=True)

    tensor_1 = tensor_1.to_gpu()

    vanilla = Attention(embedding_dim=C, n_heads=N_HEADS, context_window=T)
    vanilla.to_gpu()
    output_1 = vanilla(tensor_1)
    output_1.backward()

    with UseMixedPrecision():
        tensor_2 = tensor_2.to_gpu()
        tensor_2.array = tensor_2.array.astype(cp.float16)
        cuda = CudnnAttention(
            embedding_dim=C, n_heads=N_HEADS, context_window=T
        )
        cuda.to_gpu()
        output_2 = cuda(tensor_2)
        # output_2.backward()

    # assert output_1.close_to(output_2, rtol=1e-2, atol=1e-5)
    breakpoint()
    print(output_1)
    print(output_2)

    print(tensor_1.grad)
    print(tensor_2.grad)
    breakpoint()

    assert tensor_1.grad.close_to(tensor_2.grad, rtol=1e-2, atol=1e-5)


def generate_causal_mask(seq_length):
    import torch

    # Create a causal mask
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    return mask


def test_cudnn_attention_vs_pytorch():
    import cupy as cp
    import torch

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    cp.random.seed(42)

    # Parameters
    B, T, C = INPUT_SHAPE
    batch_size = B
    embedding_dim = C
    context_window = T
    n_heads = N_HEADS

    # Create input tensor
    input_np = np.random.randn(
        batch_size, context_window, 3 * embedding_dim
    ).astype(np.float16)
    input_tricycle = Tensor(cp.array(input_np), dtype=cp.float16).to_gpu()
    input_torch = torch.from_numpy(input_np).to(torch.float16).cuda()

    # Create attention modules
    cudnn_attention = CudnnAttention(
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        n_heads=N_HEADS,
        context_window=T,
        shared={},
    )

    # Forward pass for CudnnAttention
    with UseMixedPrecision():
        output_tricycle = cudnn_attention.forward(input_tricycle)

    torch_attention = torch.nn.MultiheadAttention(
        embed_dim=embedding_dim,
        num_heads=n_heads,
        batch_first=True,
        dtype=torch.float16,
        # bias=False,
    ).cuda()

    # Generate causal mask
    causal_mask = generate_causal_mask(context_window).cuda()

    # Forward pass for PyTorch
    mae = np.mean(
        np.abs(input_tricycle.array.get() - input_torch.cpu().detach().numpy())
    )
    print(f"Input Mean Absolute Error: {mae}")
    q, k, v = input_torch.chunk(3, dim=-1)
    output_torch, _ = torch_attention(
        q, k, v, attn_mask=causal_mask, is_causal=True
    )
    # Compare outputs
    output_tricycle_np = output_tricycle.array.get()
    output_torch_np = output_torch.cpu().detach().numpy()

    # Calculate mean absolute error
    mae = np.mean(np.abs(output_tricycle_np - output_torch_np))
    print(f"Output Mean Absolute Error: {mae}")
    breakpoint()

    # Calculate relative error
    relative_error = np.mean(
        np.abs((output_tricycle_np - output_torch_np) / output_torch_np)
    )
    print(f"Mean Relative Error: {relative_error}")

    # Check if the outputs are close enough
    tolerance = 1e-2  # Adjust this value based on your requirements
    assert (
        mae < tolerance
    ), f"Mean Absolute Error ({mae}) exceeds tolerance ({tolerance})"

    print(
        "Test passed: CudnnAttention output is close to PyTorch's MultiheadAttention output"
    )


__benchmarks__ = [
    # (bench_vanilla_gelu, bench_cuda_gelu, "handcraft kernel for gelu"),
    (bench_vanilla_relu, bench_triton_relu, "handcraft kernel for relu"),
    # (bench_vanilla_dense, bench_cuda_dense, "vanilla vs cublas matmul"),
    # (
    #     bench_vanilla_attention,
    #     bench_cudnn_attention,
    #     "vanilla vs cublas attention",
    # ),
    # (bench_vanilla_dense, bench_cuda_dense, "vanilla vs cublas matmul"),
    # (bench_cuda_dense, bench_cuda_dense, "cuda vs cuda matmul"),
    # (bench_vanilla_dense, bench_vanilla_dense, "vanilla vs vanilla matmul"),
]
if __name__ == "__main__":
    test_cudnn_attention_vs_pytorch()
