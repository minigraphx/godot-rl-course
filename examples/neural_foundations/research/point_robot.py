from __future__ import annotations

from dataclasses import dataclass, field
import math

import matplotlib.patches as patches
import numpy as np


def _array(value: np.ndarray | tuple[float, float]) -> np.ndarray:
    return np.array(value, dtype=np.float64)


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PointRobotEnv:
    start_position: np.ndarray = field(default_factory=lambda: _array((0.12, 0.5)))
    start_heading: float = 0.0
    target_position: np.ndarray = field(default_factory=lambda: _array((0.88, 0.5)))
    speed: float = 0.04
    turn_angle: float = math.radians(18.0)
    target_radius: float = 0.05
    ray_length: float = 0.35
    max_steps: int = 120

    def __post_init__(self) -> None:
        self.start_position = _array(self.start_position)
        self.target_position = _array(self.target_position)
        self.rng = np.random.default_rng(0)
        self.position = self.start_position.copy()
        self.heading = self.start_heading
        self.steps = 0
        self.previous_distance = self._distance_to_target()

    def reset(self, seed: int) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self.position = self.start_position.copy()
        self.heading = self.start_heading
        self.steps = 0
        self.previous_distance = self._distance_to_target()
        return self.observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if action not in (0, 1, 2):
            raise ValueError("action must be 0, 1, or 2")

        turn = (1 - action) * self.turn_angle
        self.heading = _wrap_angle(self.heading + turn)
        direction = np.array([math.cos(self.heading), math.sin(self.heading)])
        next_position = self.position + direction * self.speed
        self.steps += 1

        collision = bool(np.any(next_position <= 0.0) or np.any(next_position >= 1.0))
        self.position = np.clip(next_position, 0.0, 1.0)
        distance = self._distance_to_target()
        success = bool(distance <= self.target_radius)
        timeout = self.steps >= self.max_steps
        done = collision or success or timeout

        progress = self.previous_distance - distance
        reward = progress * 10.0 - 0.01
        if success:
            reward += 1.0
        if collision:
            reward -= 1.0
        self.previous_distance = distance

        return (
            self.observation(),
            float(reward),
            done,
            {
                "collision": collision,
                "success": success,
                "timeout": timeout,
                "steps": self.steps,
            },
        )

    def observation(self) -> np.ndarray:
        rays = np.array(
            [
                self._ray_distance(self.heading + self.turn_angle),
                self._ray_distance(self.heading),
                self._ray_distance(self.heading - self.turn_angle),
            ],
            dtype=np.float32,
        )
        bearing = self._target_bearing()
        return np.concatenate((rays, np.array([bearing], dtype=np.float32)))

    def render(self, axis) -> None:
        axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), aspect="equal")
        axis.add_patch(
            patches.Circle(self.target_position, self.target_radius, color="#59a14f", alpha=0.4)
        )
        axis.scatter([self.position[0]], [self.position[1]], color="#4e79a7", s=80)
        heading_vector = np.array([math.cos(self.heading), math.sin(self.heading)])
        axis.arrow(
            self.position[0],
            self.position[1],
            heading_vector[0] * 0.08,
            heading_vector[1] * 0.08,
            width=0.005,
            color="#4e79a7",
        )
        for ray_angle in (self.heading + self.turn_angle, self.heading, self.heading - self.turn_angle):
            distance = self._ray_distance(ray_angle) * self.ray_length
            axis.plot(
                [
                    self.position[0],
                    self.position[0] + math.cos(ray_angle) * distance,
                ],
                [
                    self.position[1],
                    self.position[1] + math.sin(ray_angle) * distance,
                ],
                color="#f28e2b",
                alpha=0.7,
            )

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self.target_position - self.position))

    def _target_bearing(self) -> float:
        vector = self.target_position - self.position
        target_angle = math.atan2(float(vector[1]), float(vector[0]))
        return float(_wrap_angle(target_angle - self.heading) / math.pi)

    def _ray_distance(self, angle: float) -> float:
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        distances = []
        if direction[0] > 0.0:
            distances.append((1.0 - self.position[0]) / direction[0])
        elif direction[0] < 0.0:
            distances.append((0.0 - self.position[0]) / direction[0])
        if direction[1] > 0.0:
            distances.append((1.0 - self.position[1]) / direction[1])
        elif direction[1] < 0.0:
            distances.append((0.0 - self.position[1]) / direction[1])
        distance = min(value for value in distances if value >= 0.0)
        return float(np.clip(distance / self.ray_length, 0.0, 1.0))
