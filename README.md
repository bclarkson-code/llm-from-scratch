# Tricycle

<p align="center">
    <img width="223" alt="tricycle_logo" src="https://github.com/bclarkson-code/Tricycle/assets/57139598/62405944-b27b-49bc-93c3-17ba93fc8ad7">
</p>

Ever wanted to understand how deep learning *actually* works? Tricycle is a fast,
fully functional deep learning library written from scratch in python and numpy.

Tricycle is not just a toy neural network: the file `train_smol_gpt.py` [trains
GPT-2 (124M)](#training-gpt-2-124m) on 2.3B tokens in just under 3 days on my
GPU (RTX 3090).

The entire library, from the automatic differentiation engine to GPU support to
a GPT, should be understandable to anyone with a bit of python experience. I've
tried to keep things simple without hiding any details so you should be able to
dive straight into the code and start hacking away.

Tricycle started as a thin wrapper around a numpy array and has developed into
a full deep learning framework. If you'd like to know how, check out the
[Tricycle Wiki](https://github.com/bclarkson-code/Tricycle/wiki/How-it-works).

You can also check out the Tricycle
[API documentation](https://bclarkson-code.com/Tricycle/)

## Table of Contents
- [Tricycle](#tricycle)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Features](#features)
  - [Training a GPT on Shakespeare](#training-a-gpt-on-shakespeare)
  - [Training GPT-2 (124M)](#training-gpt-2-124m)
  - [How it works](#how-it-works)
  - [What's Next?](#whats-next)
  - [Contact](#contact)

## Installation

Tricycle uses [conda](https://docs.conda.io/en/latest/) to manage dependencies.
While we do support CPU-only computation, optimisation efforts have been
focussed on GPU computation so it is pretty slow. If you do have a CUDA
capable GPU I would strongly recommend installing the gpu version of Tricycle.

If you have a CUDA capable GPU, you can install Tricycle as follows.

```bash
git clone https://github.com/bclarkson-code/Tricycle.git && cd Tricycle
conda env create -f requirements/environment.yml -n tricycle
conda activate tricycle
```

Otherwise, you can install the cpu-only version like this:

```bash
git clone https://github.com/bclarkson-code/Tricycle.git && cd Tricycle
conda env create -f requirements/environment.cpu.yml -n tricycle
conda activate tricycle
```

<details>
    <summary>Test/Dev installation</summary>
If you want to install test dependencies with GPU support you can do the following.

```bash
git clone https://github.com/bclarkson-code/Tricycle.git && cd Tricycle
conda env create -f requirements/environment.test.yml -n tricycle
conda activate tricycle
```

If you want to install test dependencies on CPU you can do the following.

```bash
git clone https://github.com/bclarkson-code/Tricycle.git && cd Tricycle
conda env create -f requirements/environment.cpu.test.yml -n tricycle
conda activate tricycle
```

Note, the test dependencies include pytorch but this is only used
as a reference to check correctness. I'm not cheating and using
pytorch for any actual computation.

</details>

## Features

Tricycle has most of the major features you'd expect in a modern deep learning
framework:

In tricycle, arrays are called `Tensor`s:

```python
from tricycle.tensor import Tensor

tensor = Tensor([1,2,3])
print(tensor) # Output: Tensor([1. 2. 3.])
```

You can do a lot of things with a tensor

```python
from tricycle.functions import Softmax

a = Tensor([1,2,3])
b = Tensor([4,5,6])

# addition
print(a + b) # Output: Tensor([5. 7. 9.], name=badd)

# comparison
print(a < b) # Output: Tensor([ True  True  True])

# more complex functions
print(Softmax()(a)) # Output: Tensor([0.09003057 0.24472848 0.66524094], name=softmax)
```

Most importantly, this includes differentiation.

```python
x = Tensor(2)

y = x ** 2 + 3 * x + 4
print(y) # Output: Tensor(14.0, name=+ 4)

# derivative of y with respect to (wrt) x is
# 2 * x + 3 = 7
y.backward() # differentiate wrt y
print(x.grad) # Output: Tensor(7.0)
```

Tricycle has many layers that you can use to build deep learning models:

```python
from tricycle.layers import Dense
from tricycle.activation import GeLU
from tricycle.blocks import MultiHeadSelfAttention

# Standard Dense/Linear layer
dense_layer = Dense(from_size=3, to_size=5)
print(dense_layer(a)) # Tensor([-1.0739521  -1.508788    0.17458707  1.578937    0.75451684], name=dense)

# GeLU nonlinearity (used in GPT-2)
c = Tensor([-2, -1, 0, 1, 2])
gelu = GeLU()
print(gelu(c)) # Tensor([-0.04540235 -0.158808    0.          0.841192    1.9545977 ], name=gelu)

# Attention
d = Tensor([[[0,1], [2,3], [4,5]]], is_batched=True)
attention = MultiHeadSelfAttention(
    embedding_dim=2,
    context_window=3,
    n_heads=1
)
print(attention(d)) # Tensor([[[ 0.15086384 -0.08797299]
                    #   [ 0.51435584 -0.39332452]
                    #   [ 0.9660988  -0.77281135]]], name=dense)
```

If you try to search for the implementation of a layer in pytorch, you'll
often find it buried under 20 different files of CUDA code. This is done in
the pursuit of raw performance, which is great, but makes learning about
how things actually work quite difficult.

In Tricycle, you can jump straight to the implementation. For example, here is
the forward pass for `LayerNorm`:

```python
def forward(self, tensor: Tensor):
    """
    Normalise a tensor along its final dimension
    """
    xp = tensor.xp
    x = tensor.array

    # Compute mean and variance along the feature dimension
    # This is pretty sensitive to errors so we need to do it at full
    # precision
    if TRICYCLE_CONTEXT.use_mixed_precision:
        x = x.astype(xp.float32)
    self._mean = x.mean(axis=-1, keepdims=True)
    self._var = x.var(axis=-1, keepdims=True)
    self._input = x

    # Normalize and scale
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
```

All of the logic in Tricycle is device-agnostic and can run on CPU or GPU
with Cupy.

```python
class UnaryMultiply(Op):
    """
    Multiply a tensor by a constant
    """
    _constant: float

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp # use the backend that grad is using.
                     # `xp` is either numpy or cupy (x is unknown)

        self._grad = xp.multiply(grad.array, self._constant)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Multiply a constant, elementwise, to a tensor. The constant is not
        differentiable.
        """
        xp = tensor.xp # use the backend that grad is using
                       # `xp` is either numpy or cupy (x is unknown)

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.multiply(tensor.array, constant)
        self._constant = constant

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name=f"+ {constant}",
            is_batched=tensor.is_batched,
        )

multiply = UnaryMultiply()
tensor = Tensor([1,2,3])

# CPU computation
print(multiply(tensor, 2)) # Tensor([2. 4. 6.], name=+ 2)

# GPU computation (this will fail if you dont have a GPU)
tensor = tensor.to_gpu()
print(multiply(tensor, 2)) # Tensor([2. 4. 6.], name=+ 2)
```

If you take away one thing from this readme, it should be to look
into the code! The best way to learn something is by getting your
hands dirty and playing around with a deep learning library that
doesn't hide anything is a great place to start.

## Training a GPT on Shakespeare

The following toy script will train a small GPT to generate convincing
Shakespeare. On my RTX 3090, this takes ~9 mins. For a more realistic
training script with metric tracking, gradient accumulation, a
validation dataset etc, take a look at `train_smol_gpt.py`

I've chosen some sensible default values for this model in
`src/tricycle/configs.py:ShakespeareConfig`. Feel free to play around with
these and see what happens. If you are running out of GPU memory, try dropping
the batch size until everything fits in memory.

If you don't have a CUDA capable GPU (or your GPU is slow), you can run the
script on CPU but it will take a while. You'll probably want to try dropping
the number of steps to something like 1000 and leave this running overnight.
The model wont be quite as convincing but it should be recognisably shakespeare.

```python
import pickle

from tqdm import tqdm

from tricycle import GPU_ENABLED
from tricycle.configs import ShakespeareConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import CrossEntropy
from tricycle.models import GPT
from tricycle.optimisers import AdamW
from tricycle_datasets.shakespeare import Shakespeare

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
```

Once trained, you can generate infinite shakespeare plays as follows:

```bash
# if you're running on CPU
python inference.py model.pkl --prompt "JULIET: Romeo, Romeo!"

# if you're running on GPU (much faster)
python inference.py --use-gpu model.pkl --prompt "JULIET: Romeo, Romeo!"
```

For the above prompt, this generated:

>  brother, his death
> name, to kill the ow of death he touched
> Exce,
> The wakes of saffords and patharing is done:
> Since this same hang'd my villain.
>
> ROMEO:
> Tybalt's that namest thy father children, stay yead in tears,
> And let me see thee take ord's shame, to again,
> If thou art trans'd, or thou stay'st cowars' say,
> O Thy  at ancivil frantic steeding chamber?

It's not perfect, but it is definitely shakespeare!

## Training GPT-2 (124M)

To train GPT-2, you'll first want to start an MLFlow server in a separate
terminal for tracking metrics during training. You can do this as follows:

```bash
mlflow server --host 0.0.0.0
```

Then you should be able to just run

```bash
python train_smol_gpt.py
```

If you have a CUDA capable GPU, this will download and tokenise the dataset
and start training.

Note, the dataset will be cached in ~/.cache/huggingface by default. You might want
to delete this once you're done because it is pretty big. Also, at present, shuffling
the dataset requires a lot of ram (I'm working on bringing this down).

The parameters for the model can be found at `src/tricycle/configs.py:SmolGPTConfig`.
If you aren't using an RTX 3090, you'll probably want to play around with the
parameters a bit to optimise things for your setup (e.g increasing batch size
if you have a GPU with more VRAM).

After training for 68 hours on my RTX 3090 (equivalent to ~2.3B tokens), the
loss curve looked like this:

![loss_curve](https://github.com/user-attachments/assets/8736c45b-b852-4c0d-b347-4092eaf6e8e8)

The best validation loss was a respectable 3.61 (perplexity of 37.0).
The resulting model certainly isn't perfect but it can produce (mostly)
coherent english.

Given the prompt "Here is my favourite limerick: there once was a man from nantucket" it responded:

> He’s so beautiful that her mother didn’t have a man who wasn’t even born a single. Now he’s not only a woman from nancy. It’s his father-in-law and his riches. I know that he’s his role. It’s a handsome man because he’s a man. It’s his mother-in-law. He’s a very, very, very good person at the same time.
> Like that character.
> All of our stories go out of our mouths.

There is still more to do (see below), but for a language model built
completely from scratch, I'd argue that this is not bad!

## What's Next?

- Documentation
  - [X] Explain how to train a language model
  - [X] Explain the tokeniser

- Code
  - [ ] Rotary Encodings (in progress)
  - [X] Test RMS Norm
  - [ ] Multi-GPU support
  - [ ] ZeRO (efficient parameter sharing between gpus)
  - [ ] Optimise and use the tokeniser (in progress)
  - [ ] 16 bit operations (in progress)
  - [ ] Pytorch <-> Tricycle/ONNX model conversions

- Experiments
  - [X] Try a language dataset rather than pure code
  - [X] Build a bigger langauge model (GPT-2 (124M))
  - [ ] Build a LLama style model (Rotary Encodings + SwiGLU + RMSNorm)

## Contact

Want to get in contact/work together? (Anthropic, HMU) You can reach me at: [bclarkson-code@proton.me](mailto:bclarkson-code@proton.me)
