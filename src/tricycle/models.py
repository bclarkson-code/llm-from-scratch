"""
GPT model implementation using the Tricycle framework.

This module defines the GPT class, which implements a GPT-style transformer model
using components from the Tricycle framework.
"""

import humanize
import numpy as np

from tricycle.blocks import GPT2TransformerBlock
from tricycle.configs import GPTConfig
from tricycle.layers import (
    Dense,
    Dropout,
    Embedding,
    Layer,
    LayerNorm,
    RMSNorm,
)
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor


class GPT(Layer):
    """
    Generative Pre-trained Transformer (GPT) model implementation.

    This class implements a GPT-style transformer model using components from
    the Tricycle framework. It includes token and position embeddings, multiple
    transformer blocks, and a final output layer.

    Attributes:
        embedding_dim (int): Dimension of the embedding space.
        context_window (int): Size of the context window for position embeddings.
        token_embedding (Embedding): Embedding layer for input tokens.
        position_embedding (Embedding): Embedding layer for positional information.
        input_dropout (Dropout): Dropout layer applied to the input embeddings.
        blocks (list): List of GPT2TransformerBlock instances.
        head (Dense): Final dense layer for output.
        norm (LayerNorm or RMSNorm): Normalization layer.
        layers (list): List of all layers in the model.
    """

    def __init__(self, config: GPTConfig):
        """
        Initializes the GPT model with the given configuration.

        Args:
            config (GPTConfig): Configuration object containing model parameters.
        """
        self.embedding_dim = config.embedding_dim
        self.context_window = config.context_window
        self.token_embedding = Embedding(
            to_size=self.embedding_dim,
            from_size=config.vocab_size,
            name="token_embedding",
        )
        self.position_embedding = Embedding(
            to_size=self.embedding_dim,
            from_size=self.context_window,
            name="position_embedding",
        )
        self.input_dropout = Dropout(config.input_dropout_prob)

        self.blocks = [
            GPT2TransformerBlock(
                embedding_dim=self.embedding_dim,
                n_heads=config.n_heads,
                context_window=self.context_window,
                expansion_ratio=config.expansion_ratio,
                activation_fn=config.activation_fn,
                norm_fn=config.norm_fn,
            )
            for _ in range(config.n_layers)
        ]

        self.head = Dense(
            to_size=config.vocab_size,
            from_size=self.embedding_dim,
            name="head",
        )
        match config.norm_fn:
            case "layer_norm":
                self.norm = LayerNorm(self.embedding_dim)
            case "rms_norm":
                self.norm = RMSNorm(self.embedding_dim)
            case _:
                raise ValueError(f"Unknown norm: {config.norm_fn}")

        self.layers = [
            self.token_embedding,
            self.position_embedding,
            self.input_dropout,
            *self.blocks,
            self.norm,
            self.head,
        ]

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Performs a forward pass through the GPT model.

        Args:
            tensor (Tensor): Input tensor, expected to be one-hot encoded.

        Returns:
            Tensor: Output tensor after passing through the model.

        Raises:
            AssertionError: If the input tensor doesn't match the expected context window size.
        """
        xp = tensor.xp
        if tensor.ndim == 1:
            n_tokens = tensor.shape[-1]
            tensor.array = xp.expand_dims(tensor.array, 0)
            tensor = tensor.to_batched()
        else:
            n_tokens = tensor.shape[-1]
        assert n_tokens == self.context_window, (
            "Expected a full context window. ",
            f"Found {n_tokens=} and {self.context_window=}",
        )

        position = Tensor(
            xp.arange(self.context_window),
            requires_grad=False,
            dtype=int,
        )

        pos_embedding = self.position_embedding(position)
        token_embedding = self.token_embedding(tensor)

        embedding = token_embedding + pos_embedding

        embedding = self.input_dropout(embedding)

        for i, block in enumerate(self.blocks):
            embedding = block(embedding)

        embedding = self.norm(embedding)

        embedding = self.head(embedding)
        return embedding

    def zero_grad(self):
        """
        Zeroes out the gradients of all layers in the model.

        Returns:
            GPT: The current GPT instance.
        """
        self.token_embedding.zero_grad()
        self.position_embedding.zero_grad()
        self.norm.zero_grad()
        self.head.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        return self

    def update(self, optimiser: Optimiser):
        """
        Updates all layers in the model using the provided optimiser.

        Args:
            optimiser (Optimiser): The optimiser to use for updating model parameters.

        Returns:
            GPT: The current GPT instance.
        """
        self.token_embedding.update(optimiser)
        self.position_embedding.update(optimiser)
        self.norm.update(optimiser)
        self.head.update(optimiser)
        for block in self.blocks:
            block.update(optimiser)
        return self

    def to_gpu(self, device: int = 0):
        """
        Moves all layers of the model to the specified GPU device.

        Args:
            device (int, optional): The GPU device number. Defaults to 0.

        Returns:
            GPT: The current GPT instance.
        """
        self.token_embedding.to_gpu(device)
        self.position_embedding.to_gpu(device)
        for block in self.blocks:
            block.to_gpu(device)
        self.norm.to_gpu(device)
        self.head.to_gpu(device)
        return self

    def from_gpu(self):
        """
        Moves all layers of the model from GPU back to CPU.

        Returns:
            GPT: The current GPT instance.
        """
        self.token_embedding.from_gpu()
        self.position_embedding.from_gpu()
        for block in self.blocks:
            block.from_gpu()
        self.norm.from_gpu()
        self.head.from_gpu()
        return self

    def display(self):
        """Prints a string representation of the model."""
        print(self)

    def _contents(self):
        """
        Returns a flattened list of the layers in this model, along with
        their depth in the tree of layers.

        Returns:
            list: A list of tuples containing layer name, size, and depth.
        """
        stack = [(self, 0)]

        contents = []
        while stack:
            node, indent = stack.pop()

            tensors = list(node.tensors.values())
            shapes = [t.shape for t in tensors]
            size = sum(np.prod(shape) for shape in shapes)
            contents.append((node.__class__.__name__, size, indent))

            stack.extend((layer, indent + 1) for layer in node.layers[::-1])
        return contents

    def __str__(self):
        """
        Returns a string representation of the model, including layer sizes
        and total parameter count.

        Returns:
            str: A formatted string representing the model structure and size.
        """
        string = ""
        total = 0
        for layer, size, n_indent in self._contents():
            total += size
            size = humanize.scientific(size) if size else ""
            indent = "  " * n_indent

            string += f"{indent}{layer}({size})\n"

        PARAM_SIZE = self.head.weights[0][0].dtype.itemsize
        total *= PARAM_SIZE

        string += "Total size:\n"
        string += f"  - {humanize.naturalsize(total)}\n"
        string += "Total parameters:\n"
        string += f"  - {humanize.intword(total/PARAM_SIZE)}\n"
        return string
