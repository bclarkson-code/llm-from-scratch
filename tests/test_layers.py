from copy import copy

import numpy as np
import pytest

from tricycle import GPU_ENABLED
from tricycle.layers import (  # noqa: E501
    CudaDense,
    Dense,
    Dropout,
    Embedding,
    LayerNorm,
    Sequential,
)
from tricycle.tensor import Tensor
from tricycle.utils import UseMixedPrecision


def test_dense_layer():
    layer = Dense(10, 8)

    assert layer.weights.shape == (10, 8)

    x_in = Tensor(np.ones(10))

    x_out = layer(x_in)
    assert x_out.shape == (8,)


def test_sequential_layer():
    layer1 = Dense(10, 8)
    layer2 = Dense(8, 4)

    model = Sequential(layer1, layer2)

    assert model.layers[0].weights.shape == (10, 8)
    assert model.layers[1].weights.shape == (8, 4)

    x_in = Tensor(np.ones(10))

    x_out = model(x_in)
    assert x_out.shape == (4,)


def test_dropout():  # sourcery skip: square-identity
    np.random.seed(0)
    size = 100
    dropout_prob = 0.3

    # non-batched
    in_tensor = Tensor(np.random.normal(size=(size, size)), name="in_tensor")
    dropout = Dropout(dropout_prob)

    out_tensor = dropout(in_tensor.to_batched())

    assert out_tensor.shape == in_tensor.shape
    zero_x_idx, zero_y_idx = np.where(out_tensor.array == 0)
    n_zeros = len(zero_x_idx)
    expected_n_zeros = int(size * size * dropout_prob)

    assert n_zeros / size**2 - expected_n_zeros / size**2 < 0.05

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.shape == in_tensor.shape

    coef = 1 / (1 - dropout_prob)
    correct_grad = np.full(in_tensor.shape, coef)
    correct_grad[zero_x_idx, zero_y_idx] = 0

    assert in_tensor.grad.close_to(correct_grad)


def test_layer_norm():
    np.random.seed(0)
    in_tensor = Tensor(np.random.normal(size=(100, 100)), name="in_tensor")
    layer_norm = LayerNorm(100)
    out_tensor = layer_norm(in_tensor.to_batched())

    assert out_tensor.shape == in_tensor.shape
    out_tensor.backward()

    assert copy(out_tensor).mean().close_to(0, atol=1e-3)
    assert np.allclose(np.std(out_tensor.array), [1] * 100, atol=1e-7)

    assert in_tensor.grad is not None
    assert in_tensor.grad.shape == in_tensor.shape

    # not sure if this is correct. TODO: check
    assert in_tensor.grad.close_to(np.zeros(in_tensor.shape), atol=1e-3)


def test_embedding():
    np.random.seed(0)
    vocab_size = 3
    out_shape = 5
    in_tensor = Tensor(
        [0, 1, 2, 0],
        requires_grad=False,
        dtype=int,
    )

    embedding_layer = Embedding(from_size=vocab_size, to_size=out_shape)
    weights = np.indices((vocab_size * out_shape,)).reshape(
        vocab_size, out_shape
    )
    embedding_layer.weights = Tensor(weights)

    result = embedding_layer(in_tensor)

    assert result.shape == (4, 5)
    assert result[0].close_to(embedding_layer.weights[0])
    assert result[1].close_to(embedding_layer.weights[1])
    assert result[2].close_to(embedding_layer.weights[2])
    assert result[3].close_to(embedding_layer.weights[0])

    result.backward()

    assert embedding_layer.weights.grad is not None
    assert embedding_layer.weights.grad.shape == embedding_layer.weights.shape
    assert embedding_layer.weights.grad.close_to(
        [[2, 2, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    )


def test_embedding_batched():
    np.random.seed(0)
    vocab_size = 3
    out_shape = 5
    in_tensor = Tensor(
        [[0, 1, 2, 0], [1, 2, 2, 1]],
        requires_grad=False,
        dtype=np.int8,
    ).to_batched()

    embedding_layer = Embedding(from_size=vocab_size, to_size=out_shape)
    weights = np.indices((vocab_size * out_shape,)).reshape(
        vocab_size, out_shape
    )
    embedding_layer.weights = Tensor(weights)

    result = embedding_layer(in_tensor)

    assert result.shape == (2, 4, 5)
    assert result[0][0].close_to(embedding_layer.weights[0])
    assert result[0][1].close_to(embedding_layer.weights[1])
    assert result[0][2].close_to(embedding_layer.weights[2])
    assert result[0][3].close_to(embedding_layer.weights[0])

    assert result[1][0].close_to(embedding_layer.weights[1])
    assert result[1][1].close_to(embedding_layer.weights[2])
    assert result[1][2].close_to(embedding_layer.weights[2])
    assert result[1][3].close_to(embedding_layer.weights[1])

    result.backward()

    assert embedding_layer.weights.grad is not None
    assert embedding_layer.weights.grad.shape == (vocab_size, out_shape)
    assert embedding_layer.weights.grad.close_to(
        [
            [
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0, 3.0],
            ],
        ]
    )


def test_dense_match():
    if not GPU_ENABLED:
        pytest.skip()

    INPUT_SHAPE = (4, 3, 2)
    OUTPUT_SHAPE = 2
    np.random.seed(0)
    import cupy as cp

    with UseMixedPrecision():
        random_data = np.random.random(INPUT_SHAPE).astype(np.float16) * 2 - 1
        tensor_1 = Tensor(random_data.copy())
        tensor_2 = Tensor(random_data.copy())

        tensor_1 = tensor_1.to_gpu()
        tensor_2 = tensor_2.to_gpu()

        dense = Dense(INPUT_SHAPE[-1], OUTPUT_SHAPE)
        dense.to_gpu()
        output_1 = dense(tensor_1)
        output_1.backward()

        cublas_dense = CudaDense(INPUT_SHAPE[-1], OUTPUT_SHAPE)
        cublas_dense.weights.array = cp.ascontiguousarray(
            dense.weights.array.copy().T, dtype=cp.float16
        )
        cublas_dense.to_gpu()
        output_2 = cublas_dense(tensor_2)
        output_2.backward()

        assert output_1.close_to(output_2, rtol=1e-2, atol=1e-5)
        assert tensor_1.grad.close_to(tensor_2.grad, rtol=1e-2, atol=1e-5)
        assert dense.weights.grad.close_to(
            cublas_dense.weights.grad, rtol=1e-2, atol=1e-5
        )
