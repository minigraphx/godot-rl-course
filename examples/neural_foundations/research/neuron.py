from collections.abc import Sequence
import math


def activate(value: float, name: str) -> float:
    if name == "identity":
        return value
    if name == "step":
        return 1.0 if value >= 0.0 else 0.0
    if name == "sigmoid":
        return 1.0 / (1.0 + math.exp(-value))
    if name == "tanh":
        return math.tanh(value)
    raise ValueError(f"Unknown activation: {name}")


def neuron_output(
    inputs: Sequence[float],
    weights: Sequence[float],
    bias: float,
    activation: str,
) -> float:
    if len(inputs) != len(weights):
        raise ValueError("inputs and weights must have equal length")
    weighted_sum = sum(value * weight for value, weight in zip(inputs, weights))
    return activate(weighted_sum + bias, activation)
