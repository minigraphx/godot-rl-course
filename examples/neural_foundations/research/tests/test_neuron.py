import unittest

from examples.neural_foundations.research.neuron import activate, neuron_output


class NeuronTests(unittest.TestCase):
    def test_weighted_sum_with_bias(self):
        self.assertAlmostEqual(
            neuron_output([0.5, -0.25], [0.8, -0.4], 0.1, "identity"),
            0.6,
        )

    def test_tanh_is_bounded(self):
        value = activate(10.0, "tanh")
        self.assertGreater(value, 0.99)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
