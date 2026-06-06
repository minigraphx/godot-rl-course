import unittest

from examples.neural_foundations.research.neuron import activate, neuron_output

SECTION_INPUTS = [0.5, -0.25]
SECTION_WEIGHTS = [0.8, -0.4]
SECTION_BIAS = 0.1


def print_walkthrough() -> None:
    contributions = [
        value * weight
        for value, weight in zip(SECTION_INPUTS, SECTION_WEIGHTS, strict=True)
    ]
    weighted_sum = neuron_output(
        SECTION_INPUTS, SECTION_WEIGHTS, SECTION_BIAS, "identity"
    )
    output = neuron_output(
        SECTION_INPUTS, SECTION_WEIGHTS, SECTION_BIAS, "tanh"
    )
    def show(line: str = "") -> None:
        print(line, flush=True)

    show("Frozen neuron (same numbers as section 1):")
    for index, (value, weight, contribution) in enumerate(
        zip(SECTION_INPUTS, SECTION_WEIGHTS, contributions, strict=True),
        start=1,
    ):
        show(
            f"  x{index}={value:.2f} × w{index}={weight:+.2f}  ->  {contribution:+.2f}"
        )
    show(f"  bias                 {SECTION_BIAS:+.2f}")
    show(f"  z = {weighted_sum:+.3f}")
    sign = "positive" if output >= 0.0 else "negative"
    show(f"  tanh(z) = {output:+.3f}  ({sign})")
    bounded = activate(10.0, "tanh")
    show()
    show("tanh stays bounded:")
    show(f"  tanh(10) = {bounded:.12f}  (never exceeds +1.0)")
    show()


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
    print_walkthrough()
    unittest.main(verbosity=2)
