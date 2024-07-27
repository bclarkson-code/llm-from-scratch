"""
Training script for SmolGPT a replication of GPT-2

The training script is pretty generic. You can tune the parameters in by
modifying the config.

Currently, we train the model on fineweb, a cleaned dump of ~10B tokens of web
data
"""

import os
import pickle
import uuid
from pathlib import Path

from tricycle import GPU_ENABLED
from tricycle.context import TRICYCLE_CONTEXT
from tricycle.ops import Op
from tricycle.tensor import Tensor
from tricycle.utils import UseMixedPrecision, optimal_n_tokens

if GPU_ENABLED:
    import cupy as xp
else:
    import numpy as xp

import mlflow
from tqdm import tqdm

from inference import get_sample
from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import CrossEntropy
from tricycle.models import GPT
from tricycle.optimisers import AdamW
from tricycle.scheduler import CosineSchedule
from tricycle_datasets.fineweb import FineWeb

# fix the seed for reproducibility
xp.random.seed(0)
config = SmolGPTConfig()


def load_datasets(config: SmolGPTConfig):
    """
    Load tokens, batch and shuffle them.
    """

    # if you are loading this for the first time, this can take a while.
    # it will create some big cache files in ~/.cache/huggingface that you
    # might want to clean up once you are done with the dataset
    print("Loading dataset")
    train_dataset = FineWeb(config.vocab_size, split="train")

    # we cant fit more than 3B indices in memory on my computer which is more
    # than we need anyway.
    # TODO: figure out how to shuffle without using much memory
    train_dataset.tokens = train_dataset.tokens[: int(3e9)]
    valid_dataset = FineWeb(config.vocab_size, split="valid")

    print("Loading dataloaders")
    train_dataloader = (
        CausalLMDataset(
            tokens=train_dataset.tokens,
            vocab_size=train_dataset.vocab_size,
            batch_size=config.batch_size,
            context_window=config.context_window,
        )
        .batch()
        .shuffle()  # only shuffle train dataset.
        .to_tensor()
    )
    valid_dataloader = (
        CausalLMDataset(
            tokens=valid_dataset.tokens,
            vocab_size=valid_dataset.vocab_size,
            batch_size=config.batch_size,
            context_window=config.context_window,
        )
        .batch()
        .to_tensor()
    )
    return (
        train_dataset,
        valid_dataset,
        train_dataloader,
        valid_dataloader,
    )


def estimate_loss(
    model: GPT,
    valid_dataloader: CausalLMDataset,
    config: SmolGPTConfig,
    loss_fn: Op,
) -> float:
    """
    Run the model on the validation dataset to estimate its loss
    """
    batch_loss = 0
    for valid_step, (inputs, outputs) in tqdm(
        enumerate(valid_dataloader), total=config.eval_steps, desc="Validation"
    ):
        if valid_step == config.eval_steps:
            break

        assert isinstance(inputs, Tensor)
        assert isinstance(outputs, Tensor)
        if GPU_ENABLED:
            inputs = inputs.to_gpu(config.device_idx)
            outputs = outputs.to_gpu(config.device_idx)

        # forward pass
        logits = model(inputs)
        loss = loss_fn(outputs, logits)
        batch_loss += loss.array / config.eval_steps

    return batch_loss


def validate(
    model: GPT,
    valid_dataset: CausalLMDataset,
    config: SmolGPTConfig,
    loss_fn: Op,
    best_loss: float,
):
    """
    Check the performance of the model on validation data. Both in terms of
    loss and by generating some sample text and storing it in MLFlow.

    If the new validation loss is better than the previous validation loss,
    save the model to disk
    """
    # generate some text
    predicted = get_sample(
        model=model,
        tokeniser=valid_dataset.tokeniser,
        sample_tokens=valid_dataset.tokens[: config.context_window],
    )
    mlflow.log_text(predicted, f"generated/{step}.txt")

    # esimate validation loss
    valid_loss = estimate_loss(
        model=model,
        valid_dataloader=valid_dataloader,
        config=config,
        loss_fn=loss_fn,
    )
    mlflow.log_metric("valid_loss", valid_loss, step=step)

    # checkpoint if new model better than old
    if valid_loss < best_loss:
        Path("models").mkdir(exist_ok=True)
        with open(f"models/model_{unique_id}.pkl", "wb") as f:
            pickle.dump(model, f)
        best_loss = valid_loss
    return best_loss


# Create our model from the config
model = GPT(config)
model.display()

# Use corrected Chinchilla scaling to estimate the compute-optimal number of
# tokens and steps we should train for
n_tokens, n_steps = optimal_n_tokens(model, config)

loss_fn = CrossEntropy()
scheduler = CosineSchedule(
    max_learning_rate=config.max_learning_rate,
    min_learning_rate=config.min_learning_rate,
    warmup_steps=config.warmup_steps,
    total_steps=n_steps,
)
optimiser = AdamW(
    learning_rate=scheduler(0),
    weight_decay=config.weight_decay,
    betas=(config.beta1, config.beta2),
)

train_dataset, valid_dataset, train_dataloader, valid_dataloader = (
    load_datasets(config)
)

if GPU_ENABLED:
    model.to_gpu(config.device_idx)


# start tracking the experiment in mlflow
mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment("SmolGPT:fineweb:base")
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
with mlflow.start_run() as run, UseMixedPrecision():
    unique_id = uuid.uuid4()

    best_loss = xp.inf

    losses = xp.zeros(n_steps)
    for step in tqdm(range(n_steps), position=0):
        mlflow.log_params(config.dict())

        optimiser.step()
        batch_loss = 0

        # perform several forward and backward passes before doing a gradient
        # update to increase the effective batch size
        for _ in range(config.gradient_accumulation_steps):
            inputs, outputs = next(train_dataloader)
            assert isinstance(inputs, Tensor)
            assert isinstance(outputs, Tensor)
            if GPU_ENABLED:
                inputs = inputs.to_gpu(config.device_idx)
                outputs = outputs.to_gpu(config.device_idx)

            # forward and backward pass
            logits = model(inputs)
            loss = loss_fn(outputs, logits)
            batch_loss += loss.array / config.gradient_accumulation_steps
            loss.backward()

        # Use the optimiser to update weights
        model.update(optimiser)

        if TRICYCLE_CONTEXT.use_mixed_precision:
            mlflow.log_metric(
                "loss",
                batch_loss / TRICYCLE_CONTEXT.loss_scale_factor,
                step=step,
            )
        else:
            mlflow.log_metric("loss", batch_loss, step=step)

        mlflow.log_metric("lr", float(optimiser.learning_rate), step=step)

        # step the learning rate
        optimiser.learning_rate = scheduler(step)

        # run validation every eval_intervals
        if step % config.eval_interval == 0:
            best_loss = validate(
                model=model,
                valid_dataset=valid_dataset,
                config=config,
                loss_fn=loss_fn,
                best_loss=best_loss,
            )

# run a final validation at the end of training
validate(
    model=model,
    valid_dataset=valid_dataset,
    config=config,
    loss_fn=loss_fn,
    best_loss=best_loss,
)
