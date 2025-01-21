import ctypes
from pathlib import Path

import cupy as cp


def attention_forward_1():
    """
    Attention with a custom cuda kernel
    """
    # dimension sizes
    batch_size = 8
    n_tokens = 32
    embedding_dim = 16
    n_heads = 2
    # initialise arrays
    cp.random.seed(0)
    input_ = cp.random.uniform(
        -5, 5, (batch_size, n_tokens, 3 * embedding_dim)
    ).astype(cp.float16)
    stats = cp.zeros(
        (batch_size, n_tokens, embedding_dim // n_heads), dtype=cp.float16
    )
    output = cp.zeros((batch_size, n_tokens, embedding_dim), dtype=cp.float16)

    # init kernel
    project_root = Path(__file__).parent
    build_dir = project_root / "build"
    lib_path = build_dir / "attention_cudnn.so"

    # Load the CUDA library
    lib = ctypes.CDLL(str(lib_path))
    lib.initialize_cuda()

    # Define argument types for the kernels
    args = [
        ctypes.c_void_p,  # float* out
        ctypes.c_void_p,  # float* stats
        ctypes.c_void_p,  # float* inp
        ctypes.c_int,  # int B
        ctypes.c_int,  # int T
        ctypes.c_int,  # int NH
        ctypes.c_int,  # int C
        ctypes.c_void_p,  # cudaStream_t stream
    ]
    kernel = lib.attention_forward_cudnn
    kernel.argtypes = args
    kernel.restype = None

    print(f"{output.shape=}")
    print(f"{stats.shape=}")
    print(f"{input_.shape=}")
    print(f"{batch_size=}")
    print(f"{n_tokens=}")
    print(f"{embedding_dim=}")
    print(f"{n_heads=}")

    # Launch function
    stream = cp.cuda.get_current_stream()
    kernel(
        ctypes.c_void_p(output.data.ptr),  # float* out
        ctypes.c_void_p(stats.data.ptr),  # float* stats
        ctypes.c_void_p(input_.data.ptr),  # float* inp
        ctypes.c_int(batch_size),  # int B
        ctypes.c_int(n_tokens),  # int T
        ctypes.c_int(n_heads),  # int NH
        ctypes.c_int(embedding_dim),  # int C
        ctypes.c_void_p(stream.ptr),
    )
    print(f"{output=}")

    # the problem is here so set a breakpoint()
    breakpoint()

    return stats, output


if __name__ == "__main__":
    attention_forward_1()
