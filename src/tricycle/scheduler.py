"""Provides learning rate scheduling functions and classes.

This module contains implementations of linear and cosine learning rate
schedules with optional warmup periods. These can be used to dynamically
adjust learning rates during training of machine learning models.

Typical usage example:

  schedule = CosineSchedule(max_learning_rate=6e-4, min_learning_rate=0,
                            total_steps=5000, warmup_steps=100)
  learning_rate = schedule(current_step)
"""

import math


def linear_schedule(
    step: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Calculates the learning rate using a linear decay schedule with warmup.

    Args:
        step: Current step in the training process.
        max_learning_rate: Maximum learning rate.
        min_learning_rate: Minimum learning rate.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of steps in the training process.

    Returns:
        The calculated learning rate for the current step.

    Raises:
        ValueError: If warmup_steps is greater than total_steps.
    """
    # avoid an off by one error
    step += 1

    if warmup_steps:
        if total_steps < warmup_steps:
            raise ValueError(
                "Cannot have a warmup longer than the total number of steps"
            )
        if step < warmup_steps:
            return (step / warmup_steps) * max_learning_rate

    coef = 1 - ((step - warmup_steps) / total_steps)
    coef *= max_learning_rate - min_learning_rate
    return min_learning_rate + coef


class CosineSchedule:
    """A class to implement a cosine decay learning rate schedule with warmup.

    Attributes:
        max_learning_rate: Maximum learning rate.
        min_learning_rate: Minimum learning rate.
        total_steps: Total number of steps in the training process.
        warmup_steps: Number of warmup steps.
        n_steps: Number of steps after warmup.
        coef: Coefficient used in the cosine decay calculation.
    """

    def __init__(
        self,
        max_learning_rate: float,
        min_learning_rate: float,
        total_steps: int,
        warmup_steps: int = 0,
    ):
        """Initialises the CosineSchedule with the given parameters.

        Args:
            max_learning_rate: Maximum learning rate.
            min_learning_rate: Minimum learning rate.
            total_steps: Total number of steps in the training process.
            warmup_steps: Number of warmup steps. Defaults to 0.

        Raises:
            ValueError: If warmup_steps is greater than total_steps.
        """
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

        if self.total_steps < self.warmup_steps:
            raise ValueError(
                "Cannot have a warmup longer than the total number of steps"
            )

        self.n_steps = total_steps - warmup_steps
        self.coef = (self.max_learning_rate - self.min_learning_rate) * 0.5

    def step(
        self,
        step: int,
    ) -> float:
        """Calculates the learning rate for a given step.

        Args:
            step: Current step in the training process.

        Returns:
            The calculated learning rate for the current step.
        """
        # use 1 indexing so our inital LR is nonzero
        step += 1

        if step < self.warmup_steps:
            return (step / self.warmup_steps) * self.max_learning_rate

        if self.warmup_steps < step < self.total_steps:
            idx = math.pi * (step - self.warmup_steps) / self.n_steps

            return self.min_learning_rate + self.coef * (math.cos(idx) + 1)

        return self.min_learning_rate

    def __call__(self, step: int) -> float:
        """Allows the class to be called as a function.

        Args:
            step: Current step in the training process.

        Returns:
            The calculated learning rate for the current step.
        """
        return self.step(step)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    x = [i for i in range(5000)]
    schedule = CosineSchedule(
        max_learning_rate=6e-4,
        min_learning_rate=0,
        total_steps=5000,
        warmup_steps=100,
    )
    y = [schedule(i) for i in x]
    plt.plot(x, y)
    plt.savefig("out.png")
