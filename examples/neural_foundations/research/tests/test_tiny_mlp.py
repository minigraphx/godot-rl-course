import unittest

import numpy as np
import torch

from examples.neural_foundations.research.tiny_mlp import TinyMLP


def fixed_network() -> TinyMLP:
    return TinyMLP(
        w_hidden=np.array([[0.2, -0.4], [0.7, 0.1]], dtype=np.float64),
        b_hidden=np.array([0.0, 0.1], dtype=np.float64),
        w_output=np.array([[0.6, -0.3]], dtype=np.float64),
        b_output=np.array([0.05], dtype=np.float64),
    )


class TinyMLPTests(unittest.TestCase):
    def test_forward_matches_numpy_reference(self):
        network = fixed_network()
        inputs = np.array([0.5, -0.25], dtype=np.float64)

        hidden_pre = np.array(
            [
                0.2 * 0.5 + -0.4 * -0.25 + 0.0,
                0.7 * 0.5 + 0.1 * -0.25 + 0.1,
            ],
            dtype=np.float64,
        )
        hidden = np.tanh(hidden_pre)
        output_pre = np.array(
            [0.6 * hidden[0] + -0.3 * hidden[1] + 0.05],
            dtype=np.float64,
        )
        expected = np.tanh(output_pre)

        np.testing.assert_allclose(network.forward(inputs), expected, atol=1e-6)

    def test_gradients_match_pytorch_autograd(self):
        network = fixed_network()
        inputs = np.array([0.5, -0.25], dtype=np.float64)
        target = np.array([0.8], dtype=np.float64)

        gradients = network.backward(inputs, target)

        torch_inputs = torch.tensor(inputs, dtype=torch.float64)
        torch_target = torch.tensor(target, dtype=torch.float64)
        w_hidden = torch.tensor(
            network.w_hidden,
            dtype=torch.float64,
            requires_grad=True,
        )
        b_hidden = torch.tensor(
            network.b_hidden,
            dtype=torch.float64,
            requires_grad=True,
        )
        w_output = torch.tensor(
            network.w_output,
            dtype=torch.float64,
            requires_grad=True,
        )
        b_output = torch.tensor(
            network.b_output,
            dtype=torch.float64,
            requires_grad=True,
        )

        hidden = torch.tanh(w_hidden @ torch_inputs + b_hidden)
        prediction = torch.tanh(w_output @ hidden + b_output)
        loss = torch.mean((prediction - torch_target) ** 2)
        loss.backward()

        np.testing.assert_allclose(gradients["w_hidden"], w_hidden.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(gradients["b_hidden"], b_hidden.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(gradients["w_output"], w_output.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(gradients["b_output"], b_output.grad.numpy(), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
