"""
Several layers can be grouped together into a single layer called a block.

This module provides various block implementations used in transformer-based
models, including multi-head self-attention, MLP blocks, and transformer blocks.
"""

from typing import Literal

import numpy as np

from tricycle.activation import GLU, GeLU, ReLU, Swish
from tricycle.attention import Attention
from tricycle.initialisers import init_xavier
from tricycle.layers import (  # noqa E501
    Dense,
    Dropout,
    Layer,
    LayerNorm,
    RMSNorm,
)
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor


def build_mask(context_window: int) -> Tensor:
    """
    Build an attention mask to stop the model from being able to see
    future tokens.

    Args:
        context_window (int): The size of the context window.

    Returns:
        Tensor: A mask tensor with shape (context_window, context_window).
    """
    NEGATIVE_INFINITY = -10_000
    mask = np.ones((context_window, context_window))
    idx = np.tril(mask.astype(bool))
    mask[~idx] = NEGATIVE_INFINITY
    mask[idx] = 0
    return Tensor(mask, requires_grad=False, name="mask")


def masked_fill(
    tensor: Tensor, mask_shape: tuple[int, int], full_mask: Tensor
):
    """
    Apply an attention_mask to a tensor.

    Args:
        tensor (Tensor): The input tensor to be masked.
        mask_shape (tuple[int, int]): The shape of the mask to be applied.
        full_mask (Tensor): The full mask tensor.

    Returns:
        Tensor: The masked tensor.
    """
    xp = tensor.xp
    repeats = tensor.shape[1] if tensor.is_batched else tensor.shape[0]
    mask = xp.stack([full_mask[: mask_shape[0], : mask_shape[1]]] * repeats)
    mask = Tensor(mask, requires_grad=False, name="mask")
    result = tensor + mask
    result.name = "masked"
    return result


class MultiHeadSelfAttention(Layer):
    """
    Multi-head self-attention layer.

    This layer implements the multi-head self-attention mechanism used in
    transformer models.

    Attributes:
        embedding_dim (int): The dimension of the input embeddings.
        n_heads (int): The number of attention heads.
        context_window (int): The size of the context window.
    """

    embedding_dim: int
    n_heads: int
    context_window: int

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
        residual_dropout_prob: float = 0.0,
        initialiser=init_xavier,
    ):
        """
        Initialize the MultiHeadSelfAttention layer.

        Args:
            embedding_dim (int): The dimension of the input embeddings.
            n_heads (int): The number of attention heads.
            context_window (int): The size of the context window.
            residual_dropout_prob (float, optional): The dropout probability for residual connections. Defaults to 0.0.
            initialiser (function, optional): The initializer function for weights. Defaults to init_xavier.
        """
        # set the constants
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.context_window = context_window

        # Project the embedding into 3 embeddings. One for each of key, query
        # and value
        self.in_projection = Dense(
            from_size=self.embedding_dim,
            to_size=self.embedding_dim * 3,
            initialiser=initialiser,
            name="in_projection",
        )

        # Pass the final embedding through a linear layer
        self.out_projection = Dense(
            from_size=self.embedding_dim,
            to_size=self.embedding_dim,
            initialiser=initialiser,
            name="out_projection",
        )

        self.residual_dropout = Dropout(residual_dropout_prob)
        self.layers = [
            self.in_projection,
            self.residual_dropout,
            self.out_projection,
        ]

        self.attention = Attention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            context_window=context_window,
        )

    def forward(self, tensor: Tensor):
        """
        Perform a forward pass through the multi-head self-attention layer.

        Args:
            tensor (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying multi-head self-attention.
        """
        # expand the input
        tensor = self.in_projection(tensor)
        attention = self.attention(tensor)

        # project back
        projected = self.out_projection(attention)
        projected = self.residual_dropout(projected)

        return projected

    def update(self, optimiser: Optimiser):
        """
        Update the layer's parameters using the given optimizer.

        Args:
            optimiser (Optimiser): The optimizer to use for updating parameters.
        """
        self.in_projection.update(optimiser)
        self.out_projection.update(optimiser)

    def zero_grad(self):
        """
        Zero out the gradients of the layer's parameters.
        """
        self.in_projection.zero_grad()
        self.out_projection.zero_grad()

    def to_gpu(self, device: int = 0):
        """
        Move the layer's parameters to the GPU.

        Args:
            device (int, optional): The GPU device number. Defaults to 0.
        """
        self.in_projection.to_gpu(device)
        self.out_projection.to_gpu(device)
        self.attention.to_gpu(device)

    def from_gpu(self):
        """
        Move the layer's parameters from the GPU to the CPU.
        """
        self.in_projection.from_gpu()
        self.out_projection.from_gpu()
        self.attention.from_gpu()


class MLPBlock(Layer):
    """
    A simple GPT-2 style MLP block with 2 linear layers around an activation
    function.

    The size of the hidden dimension is expansion_ratio * the size of the
    input.

    Attributes:
        embedding_dim (int): The dimension of the input embeddings.
        dropout_prob (float): The dropout probability.
        expansion_ratio (float): The ratio for expanding the hidden dimension.
        activation_fn (Layer): The activation function to use.
        linear_1 (Dense): The first linear layer.
        linear_2 (Dense): The second linear layer.
    """

    embedding_dim: int
    dropout_prob: float
    expansion_ratio: float
    activation_fn: Layer
    linear_1: Dense
    linear_2: Dense

    def __init__(
        self,
        embedding_dim: int,
        dropout_prob: float,
        expansion_ratio: float = 4,
        activation_fn: Layer | str = GeLU(),
    ):
        """
        Initialize the MLPBlock.

        Args:
            embedding_dim (int): The dimension of the input embeddings.
            dropout_prob (float): The dropout probability.
            expansion_ratio (float, optional): The ratio for expanding the hidden dimension. Defaults to 4.
            activation_fn (Layer | str, optional): The activation function to use. Defaults to GeLU().
        """
        self.linear_1 = Dense(
            from_size=embedding_dim,
            to_size=int(expansion_ratio * embedding_dim),
            initialiser=init_xavier,
            name="linear_1",
        )
        self.linear_2 = Dense(
            from_size=int(expansion_ratio * embedding_dim),
            to_size=embedding_dim,
            initialiser=init_xavier,
            name="linear_2",
        )
        self.dropout = Dropout(dropout_prob)
        if isinstance(activation_fn, str):
            match activation_fn:
                case "gelu":
                    activation_fn = GeLU()
                case "relu":
                    activation_fn = ReLU()
                case "swish":
                    activation_fn = Swish()
                case "glu":
                    activation_fn = GLU(int(expansion_ratio * embedding_dim))
                case _:
                    raise NotImplementedError(
                        f"Activation function {activation_fn} is not "
                        "yet implemented"
                    )
        self.activation_fn = activation_fn
        self.layers = [
            self.linear_1,
            self.activation_fn,
            self.linear_2,
            self.dropout,
        ]

    def forward(self, x: Tensor):
        """
        Perform a forward pass through the MLP block.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the MLP block.
        """
        x = self.linear_1(x)
        x = self.activation_fn(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

    def update(self, optimiser: Optimiser):
        """
        Update the layer's parameters using the given optimizer.

        Args:
            optimiser (Optimiser): The optimizer to use for updating parameters.

        Returns:
            MLPBlock: The updated MLPBlock instance.
        """
        self.linear_1.update(optimiser)
        self.linear_2.update(optimiser)
        return self

    def zero_grad(self):
        """
        Zero out the gradients of the layer's parameters.

        Returns:
            MLPBlock: The MLPBlock instance with zeroed gradients.
        """
        self.linear_1.zero_grad()
        self.linear_2.zero_grad()
        return self

    def to_gpu(self, device: int = 0):
        """
        Move the layer's parameters to the GPU.

        Args:
            device (int, optional): The GPU device number. Defaults to 0.

        Returns:
            MLPBlock: The MLPBlock instance with parameters moved to GPU.
        """
        self.linear_1.to_gpu(device)
        self.linear_2.to_gpu(device)
        return self

    def from_gpu(self):
        """
        Move the layer's parameters from the GPU to the CPU.

        Returns:
            MLPBlock: The MLPBlock instance with parameters moved to CPU.
        """
        self.linear_1.from_gpu()
        self.linear_2.from_gpu()
        return self


class GPT2TransformerBlock(Layer):
    """
    A GPT-2 style transformer block.

    This block combines multi-head self-attention with an MLP block and
    includes normalization and residual connections.

    Attributes:
        embedding_dim (int): The dimension of the input embeddings.
        expansion_ratio (float): The ratio for expanding the hidden dimension in the MLP block.
        activation_fn (Layer): The activation function to use in the MLP block.
        residual_dropout_prob (float): The dropout probability for residual connections.
        linear_dropout_prob (float): The dropout probability for the MLP block.
    """

    embedding_dim: int
    expansion_ratio: float
    activation_fn: Layer
    residual_dropout_prob: float
    linear_dropout_prob: float

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
        expansion_ratio: float = 4,
        activation_fn: Layer | str = GeLU(),
        norm_fn: Literal["layer_norm"] | Literal["rms_norm"] = "layer_norm",
        residual_dropout_prob: float = 0,
        linear_dropout_prob: float = 0,
    ):
        """
        Initialize the GPT2TransformerBlock.

        Args:
            embedding_dim (int): The dimension of the input embeddings.
            n_heads (int): The number of attention heads.
            context_window (int): The size of the context window.
            expansion_ratio (float, optional): The ratio for expanding the hidden dimension in the MLP block. Defaults to 4.
            activation_fn (Layer | str, optional): The activation function to use in the MLP block. Defaults to GeLU().
            norm_fn (Literal["layer_norm"] | Literal["rms_norm"], optional): The normalization function to use. Defaults to "layer_norm".
            residual_dropout_prob (float, optional): The dropout probability for residual connections. Defaults to 0.
            linear_dropout_prob (float, optional): The dropout probability for the MLP block. Defaults to 0.
        """
        self.attention_block = MultiHeadSelfAttention(
            embedding_dim,
            n_heads=n_heads,
            context_window=context_window,
            residual_dropout_prob=residual_dropout_prob,
            initialiser=init_xavier,
        )
        self.mlp_block = MLPBlock(
            embedding_dim,
            linear_dropout_prob,
            expansion_ratio,
            activation_fn,
        )

        match norm_fn:
            case "layer_norm":
                self.norm_1 = LayerNorm(embedding_dim)
                self.norm_2 = LayerNorm(embedding_dim)
            case "rms_norm":
                self.norm_1 = RMSNorm(embedding_dim)
                self.norm_2 = RMSNorm(embedding_dim)
            case _:
                raise ValueError(f"Unknown norm: {norm_fn}")

        self.layers = [
            self.norm_1,
            self.attention_block,
            self.norm_2,
            self.mlp_block,
        ]

    def forward(self, x: Tensor):
        """
        Perform a forward pass through the GPT-2 transformer block.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the transformer block.
        """
        normed = self.norm_1(x)

        attn = self.attention_block(normed)
        attn += x

        x = self.norm_2(attn)

        x = self.mlp_block(x)
        x += attn

        return x

    def update(self, optimiser: Optimiser):
        """
        Update the layer's parameters using the given optimizer.

        Args:
            optimiser (Optimiser): The optimizer to use for updating parameters.
        """
        self.attention_block.update(optimiser)
        self.mlp_block.update(optimiser)
        self.norm_1.update(optimiser)
        self.norm_2.update(optimiser)

    def zero_grad(self):
        """
        Zero out the gradients of the layer's parameters.
        """
        self.attention_block.zero_grad()
        self.mlp_block.zero_grad()
        self.norm_1.zero_grad()
        self.norm_2.zero_grad()

    def to_gpu(self, device: int = 0):
        """
        Move the layer's parameters to the GPU.

        Args:
            device (int, optional): The GPU device number. Defaults to 0.
        """
        self.attention_block.to_gpu(device)
        self.mlp_block.to_gpu(device)
        self.norm_1.to_gpu(device)
        self.norm_2.to_gpu(device)

    def from_gpu(self):
        """
        Move the layer's parameters from the GPU to the CPU.
        """
        self.attention_block.from_gpu()
        self.mlp_block.from_gpu()
        self.norm_1.from_gpu()
        self.norm_2.from_gpu()


class FeedForward(Layer):
    """A simple llama style feed forward block with 2 linear layers around a swiglu
    function.

    The size of the hidden dimension is expansion_ratio * the size of the
    input.

    Attributes:
        embedding_dim: The dimension of the input embedding.
        dropout_prob: The probability of dropout.
        expansion_ratio: The ratio to expand the hidden dimension.
        activation_fn: The activation function to use.
        linear_1: The first linear layer.
        linear_2: The second linear layer.

    Args:
        embedding_dim: The dimension of the input embedding.
        dropout_prob: The probability of dropout.
        expansion_ratio: The ratio to expand the hidden dimension. Defaults to 4.
        activation_fn: The activation function to use. Can be a Layer object or a string.
            Defaults to GeLU().
    """

    embedding_dim: int
    dropout_prob: float
    expansion_ratio: float
    activation_fn: Layer
    linear_1: Dense
    linear_2: Dense

    def __init__(
        self,
        embedding_dim: int,
        dropout_prob: float,
        expansion_ratio: float = 4,
        activation_fn: Layer | str = GeLU(),
    ):
        self.linear_1 = Dense(
            from_size=embedding_dim,
            to_size=int(expansion_ratio * embedding_dim),
            initialiser=init_xavier,
            name="linear_1",
        )
        self.linear_2 = Dense(
            from_size=int(expansion_ratio * embedding_dim),
            to_size=embedding_dim,
            initialiser=init_xavier,
            name="linear_2",
        )
        self.dropout = Dropout(dropout_prob)
        if isinstance(activation_fn, str):
            match activation_fn:
                case "gelu":
                    activation_fn = GeLU()
                case "relu":
                    activation_fn = ReLU()
                case "swish":
                    activation_fn = Swish()
                case "glu":
                    activation_fn = GLU(int(expansion_ratio * embedding_dim))
                case "swiglu":
                    activation_fn = SwiGLU(
                        int(expansion_ratio * embedding_dim)
                    )
                case _:
                    raise NotImplementedError(
                        f"Activation function {activation_fn} is not "
                        "yet implemented"
                    )
        self.activation_fn = activation_fn
        self.layers = [
            self.linear_1,
            self.activation_fn,
            self.linear_2,
            self.dropout,
        ]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the FeedForward layer.

        Args:
            x: Input tensor.

        Returns:
            The output tensor after passing through the feed-forward block.
        """
        x = self.linear_1(x)
        x = self.activation_fn(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

    def update(self, optimiser: Optimiser) -> "FeedForward":
        """Update the parameters of the layer using the given optimiser.

        Args:
            optimiser: The optimiser to use for updating the parameters.

        Returns:
            The updated FeedForward layer.
        """
        self.linear_1.update(optimiser)
        self.linear_2.update(optimiser)
        return self

    def zero_grad(self) -> "FeedForward":
        """Zero out the gradients of the layer's parameters.

        Returns:
            The FeedForward layer with zeroed gradients.
        """
        self.linear_1.zero_grad()
        self.linear_2.zero_grad()
        return self

    def to_gpu(self, device: int = 0) -> "FeedForward":
        """Move the layer to the GPU.

        Args:
            device: The GPU device number to move the layer to. Defaults to 0.

        Returns:
            The FeedForward layer moved to the GPU.
        """
        self.linear_1.to_gpu(device)
        self.linear_2.to_gpu(device)
        return self

    def from_gpu(self) -> "FeedForward":
        """Move the layer from the GPU to the CPU.

        Returns:
            The FeedForward layer moved to the CPU.
        """
        self.linear_1.from_gpu()
        self.linear_2.from_gpu()
        return self
