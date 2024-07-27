from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from numpy._typing import ArrayLike

from tricycle.binary import BinaryMultiply
from tricycle.context import TRICYCLE_CONTEXT
from tricycle.initialisers import init_xavier
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor
from tricycle.unary import nothing


class Layer(ABC):
    """
    A generic Layer object, representing a single operation in a neural network.

    Attributes:
        tensors (dict[str, Tensor]): Dictionary of tensors used in the layer.
        layers (Sequence[Layer]): Sequence of sub-layers, if any.
    """

    tensors: dict[str, Tensor] = {}
    layers: Sequence["Layer"] = []

    @abstractmethod
    def forward(self, tensor: Tensor):
        """
        Perform the forward pass of the layer.

        Args:
            tensor (Tensor): Input tensor.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def __call__(self, tensor: Tensor, *args, **kwargs):
        """
        Call the layer as a function.

        Args:
            tensor (Tensor): Input tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of the forward pass.
        """
        return self.forward(tensor, *args, **kwargs)

    def update(self, optimiser: Optimiser):
        """
        Update the layer's parameters using the given optimiser.

        Args:
            optimiser (Optimiser): The optimiser to use for updating parameters.
        """
        pass

    def zero_grad(self):
        """Reset gradients to zero."""
        pass

    def to_gpu(self, device: int = 0):
        """
        Move the layer to GPU.

        Args:
            device (int): The GPU device number. Defaults to 0.
        """
        pass

    def from_gpu(self):
        """Move the layer from GPU to CPU."""
        pass


class CudaLayer(Layer):
    """
    A Tricycle layer that executes CUDA code directly (as opposed to calling
    cupy methods)
    """

    cuda_folder: Path
    # TODO: Figure this out from the environment rather than hard coding
    conda_env_folder: Path = Path("/home/ben/mambaforge/envs/tricycle")
    threads_per_block: int

    def __init__(
        self,
        filename: str,
        cuda_folder: Path | None = None,
        threads_per_block: int = 512,
    ):
        import cupy as cp

        if cuda_folder is None:
            self.cuda_folder = Path(__file__).parent / "cuda"
        else:
            self.cuda_folder = cuda_folder

        kernel_path = self.cuda_folder / filename
        if not kernel_path.exists():
            raise FileNotFoundError(f"Could not find file: {kernel_path}")
        kernel_code = kernel_path.read_text()
        include_path = str(self.cuda_folder.absolute())

        # TODO: auto generate this from the environment
        system_include_paths = [
            "/usr/lib/gcc/x86_64-linux-gnu/11/include",
            "/usr/local/include",
            "/usr/include/x86_64-linux-gnu",
            "/usr/include",
            str(
                (
                    self.conda_env_folder
                    / "nsight-compute/2024.1.1/host/target-linux-x64/nvtx/include/"
                ).absolute()
            ),
            str(self.conda_env_folder / "include"),
        ]

        options = [
            "--include-path={}".format(include_path),
        ] + ["--include-path={}".format(path) for path in system_include_paths]
        # Add CUDA-specific options
        options.extend(
            [
                "--use_fast_math",
                "--restrict",
                "--std=c++11",
                "--m64",
                "-lcublas",
                "-lcublasLt",
                "-DENABLE_FP32",
                "-UENABLE_FP16",
                "-UENABLE_BF16",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "--expt-relaxed-constexpr",
            ]
        )
        self.module = cp.RawModule(
            code=kernel_code,
            options=tuple(options),
            backend="nvcc",
        )

        self.threads_per_block = threads_per_block

    def _calculate_cuda_block_size(self, input: Tensor) -> dict[str, int]:
        blocks_per_grid = (
            input.array.size + self.threads_per_block - 1
        ) // self.threads_per_block
        return {"block": (self.threads_per_block,), "grid": (blocks_per_grid,)}


class Dense(Layer):
    """
    A dense (fully connected) layer.

    Attributes:
        weights (Tensor): The weight matrix.
        from_size (int): Input size.
        to_size (int): Output size.
        name (str | None): Optional name for the layer.
    """

    weights: Tensor
    from_size: int
    to_size: int
    name: str | None

    def __init__(
        self, from_size: int, to_size: int, initialiser=init_xavier, name=None
    ):
        """
        Initialize the Dense layer.

        Args:
            from_size (int): Input size.
            to_size (int): Output size.
            initialiser: Function to initialize weights. Defaults to init_xavier.
            name (str | None): Optional name for the layer.
        """
        self.weights = initialiser(
            (from_size, to_size), name="weights" if name is None else name
        )
        self.name = name
        self.from_size = from_size
        self.to_size = to_size
        self.tensors = {"weights": self.weights}

    def weight_back_fn(self, grad: Tensor):
        """
        Compute gradients with respect to weights.

        Args:
            grad (Tensor): Gradient from the next layer.

        Returns:
            Tensor: Gradient with respect to weights.
        """
        xp = grad.xp

        indices = list(range(grad.ndim - 1))
        result = xp.tensordot(self._input, grad.array, axes=[indices, indices])
        return Tensor(
            result,
            requires_grad=grad.requires_grad,
            name="back_dense_weight",
            is_batched=False,
        )

    def grad_back_fn(self, grad: Tensor):
        """
        Compute gradients with respect to input.

        Args:
            grad (Tensor): Gradient from the next layer.

        Returns:
            Tensor: Gradient with respect to input.
        """
        xp = grad.xp

        weights = self.weights.array
        if TRICYCLE_CONTEXT.use_mixed_precision:
            weights = weights.astype(xp.float16)

        result = xp.tensordot(grad.array, self.weights.array, axes=[-1, -1])
        return Tensor(
            result,
            requires_grad=grad.requires_grad,
            name="back_dense_grad",
            is_batched=True,
        )

    def forward(self, tensor: Tensor):
        """
        Perform the forward pass of the dense layer.

        Args:
            tensor (Tensor): Input tensor.

        Returns:
            Tensor: Output of the dense layer.
        """
        xp = tensor.xp

        self._input = tensor.array
        weights = self.weights.array

        if TRICYCLE_CONTEXT.use_mixed_precision:
            weights = weights.astype(xp.float16)

        result = xp.tensordot(tensor.array, weights, axes=[-1, 0])

        return Tensor(
            result,
            name="dense",
            args=(self.weights, tensor),
            back_fns=(self.weight_back_fn, self.grad_back_fn),
            is_batched=tensor.is_batched,
        )

    def update(self, optimiser: Optimiser):
        """
        Update the weights using the given optimiser.

        Args:
            optimiser (Optimiser): The optimiser to use for updating weights.
        """
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        """Reset gradients to zero."""
        self.weights.grad = None

    def to_gpu(self, device: int = 0):
        """
        Move the layer to GPU.

        Args:
            device (int): The GPU device number. Defaults to 0.

        Returns:
            Dense: The layer itself.
        """
        self.weights.to_gpu(device)
        return self

    def from_gpu(self):
        """
        Move the layer from GPU to CPU.

        Returns:
            Dense: The layer itself.
        """
        self.weights.from_gpu()
        return self


class CudaDense(Layer):
    weights: Tensor
    from_size: int
    to_size: int
    name: str | None

    def __init__(
        self, from_size: int, to_size: int, initialiser=init_xavier, name=None
    ):
        self.weights = initialiser(
            (to_size, from_size), name="weights" if name is None else name
        )
        self.weights.array = self.weights.xp.ascontiguousarray(
            self.weights.array, dtype=self.weights.xp.float16
        )
        self.name = name
        self.from_size = from_size
        self.to_size = to_size
        self.tensors = {"weights": self.weights}

    def backward(self, grad: Tensor):
        import cupy as cp

        batch_size, n_tokens, _ = self._input.shape

        result = cp.empty(
            (batch_size, n_tokens, self.from_size), dtype=cp.float16
        )

        # calculate the gradient for the input
        llmc.matmul_input_backward(
            result,
            self.weights.array,
            grad.array,
            batch_size,
            n_tokens,
            self.from_size,
            self.to_size,
            None,
        )

        # calculate the gradient for the weights and add it to the weights.grad
        # array
        if self.weights.grad is None:
            self.weights.grad = Tensor(
                cp.zeros((self.to_size, self.from_size), dtype=cp.float16)
            )
        # this function accumulates the result rather
        # than overwriting the contents in the array
        # This lets us avoid the overhead of another op when backpropagating
        llmc.matmul_weight_backward(
            self.weights.grad.array,
            self._input,
            grad.array,
            batch_size,
            n_tokens,
            self.from_size,
            self.to_size,
            None,
        )
        return Tensor(
            result,
            requires_grad=grad.requires_grad,
            name="back_dense_grad",
            is_batched=True,
        )

    def forward(self, tensor: Tensor):
        import cupy as cp

        self._input = tensor.array
        if self._input.ndim != 3:
            raise ValueError("Only 3d arrays are supported for CudaDense")

        batch_size, n_tokens, embedding_dim = self._input.shape

        if embedding_dim != self.from_size:
            raise ValueError(
                "Expected final dimension of input to equal self.from_size . "
                f"Found {embedding_dim=} and {self.from_size=}"
            )
        result = cp.empty(
            (batch_size, n_tokens, self.to_size), dtype=cp.float16
        )

        # call the kernel. This doesn't return anything but should fill result
        # with its output instead
        llmc.matmul_forward(
            result,
            self._input,
            self.weights.array,
            batch_size,
            n_tokens,
            self.from_size,
            self.to_size,
            None,
        )

        return Tensor(
            result,
            name="dense",
            args=(tensor,),
            back_fns=(self.backward,),
            is_batched=tensor.is_batched,
        )

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self, device: int = 0):
        self.weights.to_gpu(device)
        return self

    def from_gpu(self):
        self.weights.from_gpu()
        return self


class Dropout(Layer):
    """
    A dropout layer for regularization.

    Attributes:
        probability (float): The probability of dropping out a unit.
    """

    def __init__(self, probability: float):
        """
        Initialize the Dropout layer.

        Args:
            probability (float): The probability of dropping out a unit.
        """
        self.probability = probability

    def forward(self, tensor: Tensor):
        """
        Perform the forward pass of the dropout layer.

        Args:
            tensor (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor with dropout applied.
        """
        if self.probability == 0:
            return tensor
        xp = tensor.xp
        coef = 1 / (1 - self.probability)
        random_mask = (
            xp.random.rand(*tensor.shape) > self.probability
        ).astype(tensor.dtype) * coef
        random_mask = Tensor(random_mask, is_batched=True, requires_grad=False)
        return BinaryMultiply()(tensor, random_mask)


class LayerNorm(Layer):
    """
    A Layer Normalization layer.

    Attributes:
        eps (float): A small value added for numerical stability.
        gamma (Tensor): Scale parameter.
        beta (Tensor): Shift parameter.
    """

    def __init__(self, embedding_dim: int, eps=1e-5):
        """
        Initialize the LayerNorm layer.

        Args:
            embedding_dim (int): The size of the input's last dimension.
            eps (float): A small value added for numerical stability. Defaults to 1e-5.
        """
        import numpy as np

        self.eps = eps
        self.gamma = Tensor(
            np.ones((embedding_dim,)), requires_grad=True, is_batched=False
        )
        self.beta = Tensor(
            np.zeros((embedding_dim,)), requires_grad=True, is_batched=False
        )

    def forward(self, tensor: Tensor):
        """
        Perform the forward pass of the layer normalization.

        Args:
            tensor (Tensor): Input tensor of shape (batch_size, *).

        Returns:
            Tensor: Normalized tensor of the same shape as input.
        """
        xp = tensor.xp
        x = tensor.array

        if TRICYCLE_CONTEXT.use_mixed_precision:
            x = x.astype(xp.float32)
        self._mean = x.mean(axis=-1, keepdims=True)
        self._var = x.var(axis=-1, keepdims=True)
        self._input = x

        x_norm = (x - self._mean) / xp.sqrt(self._var + self.eps)
        output = self.gamma.array * x_norm + self.beta.array

        if TRICYCLE_CONTEXT.use_mixed_precision:
            output = output.astype(xp.float16)

        return Tensor(
            output,
            is_batched=tensor.is_batched,
            requires_grad=tensor.requires_grad,
            back_fns=(self.back_fn, self.beta_back_fn, self.gamma_back_fn),
            args=(tensor, self.beta, self.gamma),
            name="layer_norm",
        )

    def gamma_back_fn(self, grad: Tensor):
        """
        Compute gradients with respect to gamma.

        Args:
            grad (Tensor): Gradient from the next layer.

        Returns:
            Tensor: Gradient with respect to gamma.
        """
        xp = grad.xp

        x_norm = (self._input - self._mean) / xp.sqrt(self._var + self.eps)
        axes = tuple(range(grad.ndim - 1))
        result = xp.sum(grad.array * x_norm, axis=axes)

        if TRICYCLE_CONTEXT.use_mixed_precision:
            result = result.astype(xp.float16)
        return Tensor(result, is_batched=False)

    def beta_back_fn(self, grad: Tensor):
        """
        Compute gradients with respect to beta.

        Args:
            grad (Tensor): Gradient from the next layer.

        Returns:
            Tensor: Gradient with respect to beta.
        """
        xp = grad.xp

        axes = tuple(range(grad.ndim - 1))
        result = xp.sum(grad.array, axis=axes)
        if TRICYCLE_CONTEXT.use_mixed_precision:
            result = result.astype(xp.float16)
        return Tensor(result, is_batched=False)

    def back_fn(self, grad: Tensor):
        """
        Compute gradients with respect to input.

        Args:
            grad (Tensor): Gradient from the next layer.

        Returns:
            Tensor: Gradient with respect to input.
        """
        xp = grad.xp

        n = self._input.shape[-1]

        dx_norm = grad.array * self.gamma.array
        dvar = xp.sum(
            dx_norm
            * (self._input - self._mean)
            * -0.5
            * xp.power(self._var + self.eps, -1.5),
            axis=-1,
            keepdims=True,
        )
        dmean = xp.sum(
            dx_norm * -1 / xp.sqrt(self._var + self.eps),
            axis=-1,
            keepdims=True,
        ) + dvar * xp.mean(
            -2 * (self._input - self._mean) / n, axis=-1, keepdims=True
        )
        result = (
            dx_norm / xp.sqrt(self._var + self.eps)
            + dvar * 2 * (self._input - self._mean) / n
            + dmean / n
        )

        if TRICYCLE_CONTEXT.use_mixed_precision:
            result = result.astype(xp.float16)
        return Tensor(
            result,
            is_batched=grad.is_batched,
            requires_grad=grad.requires_grad,
            name="back_ln",
        )

    def update(self, optimiser: Optimiser):
        """
        Update the layer's parameters using the given optimiser.

        Args:
            optimiser (Optimiser): The optimiser to use for updating parameters.
        """
        self.gamma = optimiser(self.gamma)
        self.beta = optimiser(self.beta)

    def zero_grad(self):
        """Reset gradients to zero."""
        self.gamma.grad = None
        self.beta.grad = None

    def to_gpu(self, device: int = 0):
        """
        Move the layer to GPU.

        Args:
            device (int): The GPU device number. Defaults to 0.

        Returns:
            LayerNorm: The layer itself.
        """
        self.gamma.to_gpu(device)
        self.beta.to_gpu(device)
        return self

    def from_gpu(self):
        """
        Move the layer from GPU to CPU.

        Returns:
            LayerNorm: The layer itself.
        """
        self.gamma.from_gpu()
        self.beta.from_gpu()
        return self


class RMSNorm(Layer):
    """Root Mean Square Layer Normalization.

    This class implements RMSNorm, a normalization technique that normalizes
    the inputs using the root mean square.

    Attributes:
        embedding_dim (int): The size of the input's last dimension.
        REALLY_SMALL_NUMBER (float): A small constant to avoid division by zero.
        weights (Tensor): Learnable scale parameters.

    """

    def __init__(self, embedding_dim: int, REALLY_SMALL_NUMBER=1e-4):
        import numpy as np

        self.REALLY_SMALL_NUMBER = REALLY_SMALL_NUMBER
        self.embedding_dim = embedding_dim
        self.weights = Tensor(
            np.ones((embedding_dim,)), requires_grad=True, is_batched=False
        )

    def forward(self, tensor: Tensor):
        """Applies RMS normalization to the input tensor.

        Args:
            tensor (Tensor): Input tensor to be normalized.

        Returns:
            Tensor: The normalized output tensor.
        """
        xp = tensor.xp
        x = tensor.array

        # RMSNorm is pretty sensitive to errors so we'll use full precision
        if TRICYCLE_CONTEXT.use_mixed_precision:
            x = x.astype(xp.float32)
            self.weights.array = self.weights.array.astype(xp.float32)

        # Compute square mean along the feature dimension
        mean_square = (x**2).mean(axis=-1, keepdims=True)
        self._input = x

        # Rescale
        self._divisor = 1 / xp.sqrt(mean_square) + self.REALLY_SMALL_NUMBER
        x_norm = x * self._divisor
        output = self.weights.array * x_norm

        if TRICYCLE_CONTEXT.use_mixed_precision:
            output = output.astype(xp.float16)

        return Tensor(
            output,
            is_batched=tensor.is_batched,
            requires_grad=tensor.requires_grad,
            back_fns=(self.back_fn, self.weight_back_fn),
            args=(tensor, self.weights),
            name="rms_norm",
        )

    def weight_back_fn(self, grad: Tensor):
        """Computes the gradient with respect to the weights.

        Args:
            grad (Tensor): The gradient tensor.

        Returns:
            Tensor: The gradient with respect to the weights.
        """
        xp = grad.xp

        # Compute intermediate values
        # We could have stored this but I've opted for saving memory by
        # recomputing

        x_norm = self._input * self._divisor
        axes = tuple(range(grad.ndim - 1))
        result = xp.sum(grad.array * x_norm, axis=axes)

        if TRICYCLE_CONTEXT.use_mixed_precision:
            result = result.astype(xp.float16)
        return Tensor(result, is_batched=False)

    def back_fn(self, grad: Tensor):
        """Computes the gradient with respect to the input.

        Args:
            grad (Tensor): The gradient tensor.

        Returns:
            Tensor: The gradient with respect to the input.
        """
        xp = grad.xp

        left = grad.array * self.weights.array

        right = self._input * self.weights.array
        right *= (self._divisor**2) / self.embedding_dim
        right *= xp.sum(self._input * grad.array, axis=-1, keepdims=True)

        result = left - right
        result *= self._divisor

        if TRICYCLE_CONTEXT.use_mixed_precision:
            result = result.astype(xp.float16)
        return Tensor(
            result,
            is_batched=grad.is_batched,
            requires_grad=grad.requires_grad,
            name="back_rms_norm",
        )

    def update(self, optimiser: Optimiser):
        """Updates the layer's parameters using the given optimizer.

        Args:
            optimiser (Optimiser): The optimizer to use for updating parameters.
        """
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        """Resets the gradient of the weights to None."""
        self.weights.grad = None

    def to_gpu(self, device: int = 0):
        """Moves the layer's parameters to the GPU.

        Args:
            device (int): The GPU device number.

        Returns:
            RMSNorm: The layer with parameters moved to GPU.
        """
        self.weights.to_gpu(device)
        return self

    def from_gpu(self):
        """Moves the layer's parameters from GPU to CPU.

        Returns:
            RMSNorm: The layer with parameters moved to CPU.
        """
        self.weights.from_gpu()
        return self


class Embedding(Layer):
    """Embedding layer that converts indices to dense vectors.

    This layer implements a lookup-based embedding, converting input indices
    to dense vector representations.

    Attributes:
        weights (Tensor): The embedding matrix.
        vocab_size (int): Size of the vocabulary (number of embeddings).

    """

    def __init__(
        self,
        from_size: int,
        to_size: int,
        name: str | None = None,
        initialiser=init_xavier,
    ):
        self.weights = initialiser(
            (from_size, to_size), name=name or "weights"
        )
        self.vocab_size = from_size

    def back_fn(self, grad: Tensor):
        """Computes the gradient with respect to the embedding weights.

        Args:
            grad (Tensor): The gradient tensor.

        Returns:
            Tensor: The gradient with respect to the embedding weights.
        """
        xp = grad.xp
        out = xp.zeros(self.weights.shape)

        match grad.ndim - self.input.ndim:
            case 1:
                xp.add.at(out, self.input.array, grad.array)
            case 2:
                xp.add.at(out, self.input.array, grad.array.sum(axis=0))
            case _:
                raise NotImplementedError(
                    f"{grad.ndim=}, {self.input.ndim=} are not supported"
                )

        return Tensor(out, requires_grad=grad.requires_grad)

    def forward(self, tensor: Tensor):
        """Performs the embedding lookup.

        Args:
            tensor (Tensor): Input tensor containing indices to be embedded.

        Returns:
            Tensor: The embedded representation of the input indices.
        """
        assert (
            tensor.requires_grad is False
        ), "Cannot embed a differentiable tensor"

        xp = tensor.xp
        self.input = tensor

        weights = self.weights.array
        # if were using mixed precision, copy the weights into 16 bit for
        # processing
        if TRICYCLE_CONTEXT.use_mixed_precision:
            weights = weights.astype(xp.float16)

        if tensor.is_batched:
            self._out = weights[tensor.array.flatten()].reshape(
                tensor.array.shape + (-1,)
            )
        else:
            self._out = weights[tensor.array]
        result = Tensor(self._out, is_batched=tensor.is_batched)

        result.args = (tensor, self.weights)

        result.back_fns = (nothing, self.back_fn)
        return result

    def update(self, optimiser: Optimiser):
        """Updates the embedding weights using the given optimizer.

        Args:
            optimiser (Optimiser): The optimizer to use for updating weights.
        """
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        """Resets the gradient of the weights to None."""
        self.weights.grad = None

    def to_gpu(self, device: int = 0):
        """Moves the embedding weights to the GPU.

        Args:
            device (int): The GPU device number.

        Returns:
            Embedding: The embedding layer with weights moved to GPU.
        """
        self.weights.to_gpu(device)
        return self

    def from_gpu(self):
        """Moves the embedding weights from GPU to CPU.

        Returns:
            Embedding: The embedding layer with weights moved to CPU.
        """
        self.weights.from_gpu()
        return self


class Sequential(Layer):
    """A sequential container of layers.

    This class allows for the creation of a sequential chain of layers,
    where the output of each layer is fed as input to the next layer.

    Attributes:
        layers (tuple): A tuple of Layer objects in the sequential chain.

    """

    def __init__(self, *layers: Layer):
        self.layers = layers

    def __getitem__(self, idx):
        """Retrieves a layer by index.

        Args:
            idx: The index of the layer to retrieve.

        Returns:
            Layer: The layer at the specified index.
        """
        return self.layers[idx]

    def forward(self, tensor: Tensor):
        """Performs a forward pass through all layers in the sequential chain.

        Args:
            tensor (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through all layers.
        """
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor

    def update(self, optimiser: Optimiser):
        """Updates all layers using the given optimizer.

        Args:
            optimiser (Optimiser): The optimizer to use for updating layers.
        """
        for layer in self.layers:
            layer.update(optimiser)

    def zero_grad(self):
        """Resets the gradients of all layers to None."""
        for layer in self.layers:
            layer.zero_grad()

    def to_gpu(self, device: int = 0):
        """Moves all layers to the GPU.

        Args:
            device (int): The GPU device number.
        """
        for layer in self.layers:
            layer.to_gpu(device)

    def from_gpu(self):
        """Moves all layers from GPU to CPU."""
        for layer in self.layers:
            layer.from_gpu()


class RotaryEncode(Layer):
    """Applies rotary positional encoding to a key and query.

    This layer implements the Rotary Position Embedding (RoPE) technique
    for transformer models.

    Attributes:
        embedding_dim (int): The size of the embedding dimension.
        n_heads (int): The number of attention heads.
        context_window (int): The size of the context window.
        theta (float): The base value for frequency calculation.
        head_size (int): The size of each attention head.
        freqs_cos (ArrayLike): Precomputed cosine of frequencies.
        freqs_sin (ArrayLike): Precomputed sine of frequencies.

    """

    embedding_dim: int
    n_heads: int
    context_window: int
    theta: float = 10_000.0

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
        theta: float | None = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.context_window = context_window

        self.head_size = self.context_window // self.n_heads

        if theta is not None:
            self.theta = theta

        self.freqs_cos, self.freqs_sin = self.precompute_constants()

    def precompute_constants(self) -> tuple[ArrayLike, ArrayLike]:
        """Precomputes the cosine and sine of frequencies for rotary encoding.

        Returns:
            tuple[ArrayLike, ArrayLike]: Precomputed cosine and sine values.
        """
        # this is run once at initialisation so we dont get any benefit from
        # cupy using
        import numpy as np

        # [0, 1, 2, ..., (dim//2) - 1]
        head_idx = np.arange(0, self.head_size // 2)

        # [0/dim, 2/dim, 4/dim, ... (dim-4) / dim, (dim-2) / dim]
        power = 2 * head_idx / self.head_size
        freqs = 1 / (self.theta**power)

        # assign an index to each token
        token_idx = np.arange(self.context_window)

        # this is a 2d matrix
        # freqs = t / theta**(2*d / dim)
        # where t is a token index and d is a head index
        freqs = np.outer(token_idx, freqs)

        freqs_cos = np.cos(freqs)
        freqs_sin = np.sin(freqs)

        return freqs_cos, freqs_sin

    def backward(self, grad: Tensor) -> Tensor:
        """Computes the gradient for the rotary encoding operation.

        Args:
            grad (Tensor): The gradient tensor.

        Returns:
            Tensor: The gradient with respect to the input.
        """
        xp = grad.xp

        # split the final dimension in 2 putting every
        # 2i'th value in a tensor called "grad_real"
        # and every 2i + 1'th value in a tensor called "grad_imaginary"
        grad_real = grad.array[..., 0::2]
        grad_imaginary = grad.array[..., 1::2]

        input_grad_real = (
            grad_real * self.freqs_cos + grad_imaginary * self.freqs_sin
        )
        input_grad_imaginary = (
            -grad_real * self.freqs_sin + grad_imaginary * self.freqs_cos
        )

        # Interleave the gradients back together so we get:
        # real, imaginary, real, imaginary, ...
        out = xp.empty(grad.shape)
        out[..., 0::2] = input_grad_real
        out[..., 1::2] = input_grad_imaginary

        return Tensor(
            array=out,
            name="back_rotary_encode",
            is_batched=grad.is_batched,
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """Applies rotary positional encoding to the input tensor.

        Args:
            tensor (Tensor): The input tensor.

        Returns:
            Tensor: The tensor with rotary positional encoding applied.
        """
        xp = tensor.xp

        # split the final dimension in 2 putting every
        # 2i'th value an a tensor called "real"
        # and every 2i + 1'th value in a tensor called "imaginary"
        real = tensor.array[..., 0::2]
        imaginary = tensor.array[..., 1::2]

        # combine the real an imaginary parts together with frequencies
        out_real = real * self.freqs_cos - imaginary * self.freqs_sin
        out_imaginary = real * self.freqs_sin + imaginary * self.freqs_cos

        # Interleave the real and imaginary parts
        # back together so we get:
        # real, imaginary, real, imaginary, ...
        # in the final dimension
        out = xp.empty(tensor.shape)
        out[..., 0::2] = out_real
        out[..., 1::2] = out_imaginary

        return Tensor(
            array=out,
            args=(tensor,),
            back_fns=(self.backward,),
            name="rotary_encode",
            is_batched=tensor.is_batched,
        )
