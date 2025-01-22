import pickle

from tqdm import tqdm

from tricycle import GPU_ENABLED
from tricycle.configs import ShakespeareConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import CrossEntropy
from tricycle.models import GPT
from tricycle.optimisers import AdamW
from tricycle.tricycle_datasets.shakespeare import Shakespeare

config = ShakespeareConfig()

# drop this down to 1000 for a worse model that trains faster
config.steps = 5000

model = GPT(config)

tokens = Shakespeare(vocab_size=config.vocab_size)
dataset = (
    CausalLMDataset(
        tokens=tokens,
        vocab_size=config.vocab_size,
        batch_size=config.batch_size,
        context_window=config.context_window,
    )
    .batch()
    .shuffle()
    .to_tensor()
)
loss_fn = CrossEntropy()
optimiser = AdamW(
    learning_rate=config.max_learning_rate,
    weight_decay=config.weight_decay,
    betas=(config.beta1, config.beta2),
)

if GPU_ENABLED:
    dataset = dataset.to_gpu()
    model.to_gpu()

loading_bar = tqdm(range(config.steps))
for step in loading_bar:
    optimiser.step()
    inputs, outputs = next(dataset)

    logits = model(inputs)
    loss = loss_fn(outputs, logits)
    loss.backward()

    loading_bar.set_description(f"loss: {loss.numpy().item():.3f}")
    model.update(optimiser)

# save results
with open("model.pkl", "wb") as f:
    if GPU_ENABLED:
        model.from_gpu()
    model = model.zero_grad()
    pickle.dump(model, f)
