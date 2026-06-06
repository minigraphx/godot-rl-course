import unittest

import numpy as np

from examples.neural_foundations.research.point_robot import PointRobotEnv


class PointRobotEnvTests(unittest.TestCase):
    def test_reset_restores_declared_start_state(self):
        env = PointRobotEnv()
        env.reset(seed=7)
        env.step(1)

        observation = env.reset(seed=7)

        np.testing.assert_allclose(env.position, env.start_position)
        self.assertAlmostEqual(env.heading, env.start_heading)
        np.testing.assert_allclose(observation, env.observation())

    def test_forward_action_moves_by_declared_speed(self):
        env = PointRobotEnv()
        env.reset(seed=1)
        start = env.position.copy()

        env.step(1)

        distance = np.linalg.norm(env.position - start)
        self.assertAlmostEqual(distance, env.speed, places=6)

    def test_left_and_right_actions_change_heading_symmetrically(self):
        left_env = PointRobotEnv()
        right_env = PointRobotEnv()
        left_env.reset(seed=3)
        right_env.reset(seed=3)

        left_env.step(0)
        right_env.step(2)

        self.assertAlmostEqual(
            left_env.heading - left_env.start_heading,
            -(right_env.heading - right_env.start_heading),
        )

    def test_ray_distances_are_normalized(self):
        env = PointRobotEnv()
        observation = env.reset(seed=5)

        self.assertTrue(np.all(observation[:3] >= 0.0))
        self.assertTrue(np.all(observation[:3] <= 1.0))

    def test_collision_ends_episode(self):
        env = PointRobotEnv(start_position=np.array([0.96, 0.5]), start_heading=0.0)
        env.reset(seed=11)

        _observation, _reward, done, info = env.step(1)

        self.assertTrue(done)
        self.assertTrue(info["collision"])

    def test_collision_still_returns_normalized_observation(self):
        env = PointRobotEnv(start_position=np.array([0.99, 0.5]), start_heading=0.0)
        env.reset(seed=11)

        observation, _reward, done, info = env.step(1)

        self.assertTrue(done)
        self.assertTrue(info["collision"])
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertTrue(np.all(observation[:3] >= 0.0))
        self.assertTrue(np.all(observation[:3] <= 1.0))

    def test_reaching_target_sets_success(self):
        env = PointRobotEnv(start_position=np.array([0.88, 0.5]), start_heading=0.0)
        env.reset(seed=13)

        _observation, _reward, done, info = env.step(1)

        self.assertTrue(done)
        self.assertTrue(info["success"])


if __name__ == "__main__":
    unittest.main()
