from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TinyMLP:
    w_hidden: np.ndarray
    b_hidden: np.ndarray
    w_output: np.ndarray
    b_output: np.ndarray

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        hidden_pre = self.w_hidden @ inputs + self.b_hidden
        hidden = np.tanh(hidden_pre)
        output_pre = self.w_output @ hidden + self.b_output
        return np.tanh(output_pre)

    def loss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((prediction - target) ** 2))

    def backward(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, np.ndarray]:
        hidden_pre = self.w_hidden @ inputs + self.b_hidden
        hidden = np.tanh(hidden_pre)
        output_pre = self.w_output @ hidden + self.b_output
        prediction = np.tanh(output_pre)

        d_prediction = 2.0 * (prediction - target) / target.size
        d_output_pre = d_prediction * (1.0 - prediction**2)
        d_w_output = np.outer(d_output_pre, hidden)
        d_b_output = d_output_pre.copy()

        d_hidden = self.w_output.T @ d_output_pre
        d_hidden_pre = d_hidden * (1.0 - hidden**2)
        d_w_hidden = np.outer(d_hidden_pre, inputs)
        d_b_hidden = d_hidden_pre.copy()

        return {
            "w_hidden": d_w_hidden,
            "b_hidden": d_b_hidden,
            "w_output": d_w_output,
            "b_output": d_b_output,
        }

    def apply_gradients(
        self,
        gradients: dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        self.w_hidden -= learning_rate * gradients["w_hidden"]
        self.b_hidden -= learning_rate * gradients["b_hidden"]
        self.w_output -= learning_rate * gradients["w_output"]
        self.b_output -= learning_rate * gradients["b_output"]
