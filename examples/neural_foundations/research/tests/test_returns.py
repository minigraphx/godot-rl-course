import unittest

import torch

from examples.neural_foundations.research.reinforce import discounted_returns


class DiscountedReturnsTests(unittest.TestCase):
    def test_discounted_returns_accumulate_future_reward(self):
        returns = discounted_returns([1.0, 2.0, 3.0], gamma=0.5)

        torch.testing.assert_close(
            returns,
            torch.tensor([2.75, 3.5, 3.0], dtype=torch.float32),
        )


if __name__ == "__main__":
    unittest.main()
