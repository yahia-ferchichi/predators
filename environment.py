"""
2D Grid Environment for Predator-Prey Simulation
"""

import numpy as np
from typing import Tuple, List, Optional


class Environment:
    """
    A 2D grid world where predators chase prey.

    The environment handles:
    - Agent positions
    - Movement validation
    - State representation for Q-learning
    - Episode termination conditions
    """

    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.predator_pos: Optional[Tuple[int, int]] = None
        self.prey_pos: Optional[Tuple[int, int]] = None
        self.steps = 0
        self.max_steps = 200  # Episode timeout

    def reset(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Reset the environment for a new episode.
        Places predator and prey at random positions with minimum distance.
        """
        self.steps = 0

        # Place predator randomly
        self.predator_pos = (
            np.random.randint(0, self.width),
            np.random.randint(0, self.height)
        )

        # Place prey at least 5 cells away from predator
        while True:
            self.prey_pos = (
                np.random.randint(0, self.width),
                np.random.randint(0, self.height)
            )
            if self._get_distance() >= 5:
                break

        return self.predator_pos, self.prey_pos

    def _get_distance(self) -> float:
        """Calculate Manhattan distance between predator and prey."""
        return abs(self.predator_pos[0] - self.prey_pos[0]) + \
               abs(self.predator_pos[1] - self.prey_pos[1])

    def get_relative_state(self, is_predator: bool) -> Tuple[int, int]:
        """
        Get state as relative position (discretized for Q-table).
        Returns direction to target: (dx_sign, dy_sign)
        """
        dx = self.prey_pos[0] - self.predator_pos[0]
        dy = self.prey_pos[1] - self.predator_pos[1]

        # Discretize to -1, 0, +1 for each axis
        dx_sign = 0 if dx == 0 else (1 if dx > 0 else -1)
        dy_sign = 0 if dy == 0 else (1 if dy > 0 else -1)

        if is_predator:
            # Predator sees direction to prey
            return (dx_sign + 1, dy_sign + 1)  # Shift to 0,1,2 for Q-table indexing
        else:
            # Prey sees direction away from predator (inverted)
            return (-dx_sign + 1, -dy_sign + 1)

    def get_distance_state(self) -> int:
        """
        Get discretized distance state for more nuanced learning.
        Returns: 0 (very close), 1 (close), 2 (medium), 3 (far)
        """
        dist = self._get_distance()
        if dist <= 2:
            return 0
        elif dist <= 5:
            return 1
        elif dist <= 10:
            return 2
        else:
            return 3

    def move_agent(self, is_predator: bool, action: int) -> Tuple[int, int]:
        """
        Move an agent based on action.
        Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
        Returns new position.
        """
        if is_predator:
            pos = self.predator_pos
        else:
            pos = self.prey_pos

        # Action to movement mapping
        moves = {
            0: (0, -1),   # Up
            1: (0, 1),    # Down
            2: (-1, 0),   # Left
            3: (1, 0),    # Right
            4: (0, 0)     # Stay
        }

        dx, dy = moves.get(action, (0, 0))
        new_x = max(0, min(self.width - 1, pos[0] + dx))
        new_y = max(0, min(self.height - 1, pos[1] + dy))
        new_pos = (new_x, new_y)

        if is_predator:
            self.predator_pos = new_pos
        else:
            self.prey_pos = new_pos

        return new_pos

    def step(self, predator_action: int, prey_action: int) -> Tuple[float, float, bool]:
        """
        Execute one step: both agents move, calculate rewards.

        Returns:
            predator_reward: +10 catch, -0.1 per step, bonus for getting closer
            prey_reward: +0.1 survival, -10 caught, bonus for escaping
            done: True if caught or timeout
        """
        self.steps += 1

        old_distance = self._get_distance()

        # Move both agents
        self.move_agent(True, predator_action)   # Predator
        self.move_agent(False, prey_action)       # Prey

        new_distance = self._get_distance()

        # Check if caught
        caught = self.predator_pos == self.prey_pos
        timeout = self.steps >= self.max_steps
        done = caught or timeout

        # Calculate rewards
        if caught:
            predator_reward = 10.0
            prey_reward = -10.0
        else:
            # Predator: reward for getting closer
            distance_delta = old_distance - new_distance
            predator_reward = distance_delta * 0.5 - 0.1  # Small step penalty

            # Prey: reward for escaping
            prey_reward = -distance_delta * 0.5 + 0.1  # Small survival bonus

            if timeout:
                prey_reward += 5.0  # Prey survived the episode!
                predator_reward -= 5.0

        return predator_reward, prey_reward, done

    def render_grid(self) -> np.ndarray:
        """
        Create a visual representation of the grid.
        Returns RGB array for visualization.
        """
        # Create RGB grid (height, width, 3)
        grid = np.ones((self.height, self.width, 3)) * 0.15  # Dark background

        # Add subtle grid pattern
        for i in range(self.height):
            for j in range(self.width):
                if (i + j) % 2 == 0:
                    grid[i, j] = [0.12, 0.15, 0.12]  # Subtle green tint
                else:
                    grid[i, j] = [0.1, 0.12, 0.1]

        return grid
