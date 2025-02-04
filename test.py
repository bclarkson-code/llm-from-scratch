import numpy as np
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
from tricycle.tensor import Tensor


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "tricycle"],  # Possible values for `line_arg`.
        line_names=["Triton", "Tricycle"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-relu-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = np.random.random(size).astype(np.float32)
    x = Tensor(x).to_gpu()
    y = np.random.random(size).astype(np.float32)
    y = Tensor(y).to_gpu()
    quantiles = [0.5, 0.2, 0.8]
    if provider == "tricycle":
        layer = GeLU()

    if provider == "triton":
        layer = TritonGeLU()

    def fn():
        layer(x)
        layer.backward(y)

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(), quantiles=quantiles
    )
    gbps = lambda ms: 3 * x.array.size * 32 * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)
plt.savefig("fig.png")
