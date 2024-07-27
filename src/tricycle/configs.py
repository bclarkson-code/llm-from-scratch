"""Configurations for different GPT models.

This module contains configuration classes for various GPT models, including
a base configuration class and specific configurations for debugging,
Shakespeare-based models, and a small GPT model.

Classes:
    GPTConfig: Base configuration class for GPT models.
    DebugConfig: Configuration for debugging purposes.
    ShakespeareConfig: Configuration for Shakespeare-based models.
    SmolGPTConfig: Configuration for a small GPT model.
"""

from typing import Literal


class GPTConfig:
    """Base configuration class for GPT models.

    This class defines the common parameters and hyperparameters used in
    GPT model training and evaluation.

    Attributes:
        embedding_dim (int): Dimension of the embedding layer.
        context_window (int): Size of the context window.
        vocab_size (int): Size of the vocabulary.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of transformer layers.
        expansion_ratio (float): Expansion ratio for feed-forward layers.
        activation_fn (str): Activation function used in the model.
        norm_fn (str): Normalization function used in the model.
        input_dropout_prob (float): Dropout probability for input embeddings.
        residual_dropout_prob (float): Dropout probability for residual connections.
        linear_dropout_prob (float): Dropout probability for linear layers.
        max_learning_rate (float): Maximum learning rate for training.
        min_learning_rate (float): Minimum learning rate for training.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        weight_decay (float): Weight decay factor for regularization.
        momentum (float): Momentum factor for optimization.
        beta1 (float): Beta1 parameter for Adam optimizer.
        beta2 (float): Beta2 parameter for Adam optimizer.
        steps (int | Literal["chinchilla_optimal"]): Number of training steps or "chinchilla_optimal".
        eval_interval (int): Interval between evaluations.
        batch_size (int): Batch size for training.
        gradient_accumulation_steps (int): Number of steps for gradient accumulation.
        device_idx (int): Index of the device to use for training.
        mlflow_tracking_uri (str): URI for MLflow tracking server.
        mlflow_experiment_name (str): Name of the MLflow experiment.
    """

    embedding_dim: int
    context_window: int
    vocab_size: int
    n_heads: int
    n_layers: int
    expansion_ratio: float
    activation_fn: str
    norm_fn: str

    input_dropout_prob: float
    residual_dropout_prob: float
    linear_dropout_prob: float

    max_learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    momentum = float
    beta1: float
    beta2: float

    steps: int | Literal["chinchilla_optimal"]
    eval_interval: int
    batch_size: int
    gradient_accumulation_steps: int

    device_idx: int

    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    def dict(self) -> dict[str, int | float | str | bool]:
        """Convert the configuration to a dictionary.

        Returns:
            dict[str, int | float | str | bool]: A dictionary representation of the configuration.
        """
        out = {}
        for k, v in self.__class__.__dict__.items():
            if k.startswith("__"):
                continue

            if callable(v):
                continue
            out[k] = v
        return out


class DebugConfig(GPTConfig):
    """Configuration for debugging purposes.

    This class inherits from GPTConfig and sets specific values for debugging.
    """

    embedding_dim = 14
    context_window = 13
    vocab_size = 11
    n_heads = 2
    n_layers = 1
    expansion_ratio = 4
    activation_fn = "gelu"
    norm_fn = "layer_norm"

    input_dropout_prob = 0.2
    residual_dropout_prob = 0.2
    linear_dropout_prob = 0.2

    max_learning_rate = 1e-3
    min_learning_rate = 1e-4
    warmup_steps = 100
    weight_decay = 1e-1
    momentum = 0
    beta1 = 0.9
    beta2 = 0.99

    steps = 250
    eval_interval = 1
    eval_steps = 1
    batch_size = 5
    gradient_accumulation_steps = 1
    sample_size = 4

    device_idx = 0

    mlflow_enabled = False
    mlflow_tracking_uri = ""


class ShakespeareConfig(GPTConfig):
    """Configuration for Shakespeare-based models.

    This class inherits from GPTConfig and sets specific values for
    Shakespeare-based language models.
    """

    embedding_dim = 384
    context_window = 256
    vocab_size = 1024
    n_heads = 6
    n_layers = 6
    expansion_ratio = 4
    activation_fn = "gelu"
    norm_fn = "layer_norm"

    input_dropout_prob = 0.2
    residual_dropout_prob = 0.2
    linear_dropout_prob = 0.2

    max_learning_rate = 1e-2
    min_learning_rate = 1e-4
    warmup_steps = 100
    weight_decay = 1e-1
    momentum = 0
    beta1 = 0.9
    beta2 = 0.99

    steps = 5000
    eval_interval = 250
    eval_steps = 128
    batch_size = 128
    gradient_accumulation_steps = 1
    sample_size = 512

    device_idx = 1

    mlflow_enabled = True
    mlflow_tracking_uri = "http://localhost:5000"


class SmolGPTConfig(GPTConfig):
    """Configuration for a small GPT model.

    This class inherits from GPTConfig and sets specific values for
    a small-scale GPT model.
    """

    embedding_dim = 768
    context_window = 1024
    vocab_size = 50256
    n_heads = 12
    n_layers = 12
    expansion_ratio = 4
    activation_fn = "gelu"
    norm_fn = "layer_norm"

    input_dropout_prob = 0
    residual_dropout_prob = 0
    linear_dropout_prob = 0

    max_learning_rate = 6e-4
    min_learning_rate = 0
    warmup_steps = 150  # roughly matches andrej's warmup steps in llm.c
    weight_decay = 1e-1
    momentum = 0
    beta1 = 0.9
    beta2 = 0.95

    steps = "chinchilla_optimal"
    eval_interval = 100
    eval_steps = 128
    batch_size = 4
    gradient_accumulation_steps = 128  # effective batch size of 524288 tokens
    n_tokens_to_generate = 512

    device_idx = 0

    mlflow_enabled = True
    mlflow_tracking_uri = "http://localhost:5000"
