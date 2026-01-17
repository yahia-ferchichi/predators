"""
Utility functions for Predator-Prey Simulation
"""

import numpy as np
from typing import List, Tuple
import os


def moving_average(data: List[float], window: int = 50) -> np.ndarray:
    """Calculate moving average of data."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def get_state(env, is_predator: bool) -> Tuple[int, int, int]:
    """
    Get complete state tuple for Q-learning.
    Combines relative position and distance information.
    """
    rel_state = env.get_relative_state(is_predator)
    dist_state = env.get_distance_state()
    return (rel_state[0], rel_state[1], dist_state)


def action_to_string(action: int) -> str:
    """Convert action number to readable string."""
    actions = {
        0: "↑ Up",
        1: "↓ Down",
        2: "← Left",
        3: "→ Right",
        4: "● Stay"
    }
    return actions.get(action, "Unknown")


def print_episode_stats(episode: int, predator_stats: dict, prey_stats: dict, catches: int):
    """Print formatted episode statistics."""
    print(f"\n{'='*50}")
    print(f"Episode {episode}")
    print(f"{'='*50}")
    print(f"Predator - ε: {predator_stats['epsilon']:.3f}, "
          f"Avg Reward: {predator_stats['avg_reward_last_100']:.2f}")
    print(f"Prey     - ε: {prey_stats['epsilon']:.3f}, "
          f"Avg Reward: {prey_stats['avg_reward_last_100']:.2f}")
    print(f"Catches this session: {catches}")


def ensure_dir(path: str):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def interpolate_color(color1: Tuple, color2: Tuple, t: float) -> Tuple:
    """Interpolate between two RGB colors."""
    return tuple(
        color1[i] + (color2[i] - color1[i]) * t
        for i in range(3)
    )


def create_gradient_circle(size: int, color: Tuple[float, float, float]) -> np.ndarray:
    """
    Create a gradient circle for smooth agent rendering.
    Returns RGBA array.
    """
    circle = np.zeros((size, size, 4))
    center = size // 2

    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if dist <= center:
                # Smooth falloff
                alpha = 1.0 - (dist / center) ** 0.5
                intensity = 1.0 - (dist / center) ** 2
                circle[y, x, 0] = color[0] * intensity
                circle[y, x, 1] = color[1] * intensity
                circle[y, x, 2] = color[2] * intensity
                circle[y, x, 3] = alpha

    return circle


class RewardTracker:
    """Track and analyze rewards over episodes."""

    def __init__(self):
        self.predator_rewards: List[float] = []
        self.prey_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.catches: List[bool] = []

    def add_episode(self, pred_reward: float, prey_reward: float,
                    length: int, caught: bool):
        """Record episode data."""
        self.predator_rewards.append(pred_reward)
        self.prey_rewards.append(prey_reward)
        self.episode_lengths.append(length)
        self.catches.append(caught)

    def get_catch_rate(self, last_n: int = 100) -> float:
        """Get catch rate over last n episodes."""
        recent = self.catches[-last_n:] if self.catches else []
        return sum(recent) / len(recent) if recent else 0.0

    def get_avg_length(self, last_n: int = 100) -> float:
        """Get average episode length over last n episodes."""
        recent = self.episode_lengths[-last_n:] if self.episode_lengths else []
        return np.mean(recent) if recent else 0.0
