"""
Ocean Environment - Fish escaping from a sweeping Shark
========================================================
Gym-compatible environment for Deep RL training.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Dict, Optional
import math


class OceanEnvironment(gym.Env):
    """
    A 2D ocean where fish learn to escape from a sweeping shark.
    
    - Shark: Large predator that sweeps horizontally from bottom to top
    - Fish: Small agents that share a policy network, learn to survive
    - Observation: Raycast-based detection of shark and walls
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        num_fish: int = 15,
        num_rays: int = 24,
        ray_length: float = 150.0,
        shark_speed: float = 2.0,
        fish_speed: float = 4.0,
        shark_width: float = 120.0,
        shark_height: float = 60.0,
        fish_radius: float = 8.0,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.width = width
        self.height = height
        self.num_fish = num_fish
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.shark_speed = shark_speed
        self.fish_speed = fish_speed
        self.shark_width = shark_width
        self.shark_height = shark_height
        self.fish_radius = fish_radius
        self.render_mode = render_mode
        
        # Observation: raycast distances (normalized) + shark relative position
        # Each ray returns distance to shark (0-1) and distance to wall (0-1)
        # Plus: shark direction (2), shark distance (1), velocity (2)
        self.obs_dim = num_rays * 2 + 5
        
        # Action: continuous movement direction (dx, dy) normalized
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # State
        self.fish_positions: np.ndarray = None
        self.fish_velocities: np.ndarray = None
        self.fish_alive: np.ndarray = None
        self.shark_pos: np.ndarray = None
        self.shark_direction: int = 1  # 1 = right, -1 = left
        self.steps = 0
        self.max_steps = 1000
        
        # Stats
        self.total_survived = 0
        self.total_eaten = 0
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.steps = 0
        
        # Initialize shark at bottom, moving right
        self.shark_pos = np.array([self.width / 2, self.shark_height / 2 + 20], dtype=np.float32)
        self.shark_direction = 1
        
        # Initialize fish randomly in upper 2/3 of screen
        self.fish_positions = np.zeros((self.num_fish, 2), dtype=np.float32)
        self.fish_velocities = np.zeros((self.num_fish, 2), dtype=np.float32)
        self.fish_alive = np.ones(self.num_fish, dtype=bool)
        
        for i in range(self.num_fish):
            self.fish_positions[i] = [
                np.random.uniform(50, self.width - 50),
                np.random.uniform(self.height * 0.4, self.height - 50)
            ]
            # Random initial velocity
            angle = np.random.uniform(0, 2 * np.pi)
            self.fish_velocities[i] = [np.cos(angle) * 0.5, np.sin(angle) * 0.5]
        
        # Return observation for first fish (they share the network)
        obs = self._get_observation(0)
        return obs, {}
    
    def _get_observation(self, fish_idx: int) -> np.ndarray:
        """
        Get raycast-based observation for a fish.
        """
        if not self.fish_alive[fish_idx]:
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        pos = self.fish_positions[fish_idx]
        vel = self.fish_velocities[fish_idx]
        
        obs = []
        
        # Raycast in all directions
        for i in range(self.num_rays):
            angle = (2 * np.pi * i) / self.num_rays
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            
            # Distance to shark (normalized)
            shark_dist = self._raycast_shark(pos, ray_dir)
            obs.append(shark_dist / self.ray_length)
            
            # Distance to wall (normalized)
            wall_dist = self._raycast_wall(pos, ray_dir)
            obs.append(wall_dist / self.ray_length)
        
        # Shark relative direction (normalized)
        to_shark = self.shark_pos - pos
        dist_to_shark = np.linalg.norm(to_shark) + 1e-6
        obs.append(to_shark[0] / self.width)   # Relative x
        obs.append(to_shark[1] / self.height)  # Relative y
        obs.append(min(dist_to_shark / 300.0, 1.0))  # Distance (capped)
        
        # Current velocity (normalized)
        obs.append(vel[0] / self.fish_speed)
        obs.append(vel[1] / self.fish_speed)
        
        return np.array(obs, dtype=np.float32).clip(-1, 1)
    
    def _raycast_shark(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """
        Cast a ray and return distance to shark (or ray_length if no hit).
        Simplified: check if ray intersects shark's bounding box.
        """
        # Shark bounding box
        shark_left = self.shark_pos[0] - self.shark_width / 2
        shark_right = self.shark_pos[0] + self.shark_width / 2
        shark_bottom = self.shark_pos[1] - self.shark_height / 2
        shark_top = self.shark_pos[1] + self.shark_height / 2
        
        # Ray-box intersection
        t_min = 0.0
        t_max = self.ray_length
        
        for axis in range(2):
            if abs(direction[axis]) < 1e-6:
                # Ray parallel to axis
                if axis == 0:
                    if origin[0] < shark_left or origin[0] > shark_right:
                        return self.ray_length
                else:
                    if origin[1] < shark_bottom or origin[1] > shark_top:
                        return self.ray_length
            else:
                if axis == 0:
                    t1 = (shark_left - origin[0]) / direction[0]
                    t2 = (shark_right - origin[0]) / direction[0]
                else:
                    t1 = (shark_bottom - origin[1]) / direction[1]
                    t2 = (shark_top - origin[1]) / direction[1]
                
                if t1 > t2:
                    t1, t2 = t2, t1
                
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                
                if t_min > t_max:
                    return self.ray_length
        
        if t_min > 0:
            return t_min
        return self.ray_length
    
    def _raycast_wall(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """
        Cast a ray and return distance to nearest wall.
        """
        min_dist = self.ray_length
        
        # Check all four walls
        walls = [
            (0, origin[0], -direction[0]),           # Left wall
            (self.width, self.width - origin[0], direction[0]),    # Right wall
            (0, origin[1], -direction[1]),           # Bottom wall
            (self.height, self.height - origin[1], direction[1])   # Top wall
        ]
        
        for _, dist_to_wall, dir_component in walls:
            if dir_component > 1e-6:
                t = dist_to_wall / dir_component
                if 0 < t < min_dist:
                    min_dist = t
        
        return min_dist
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step all fish with the same action pattern (shared policy).
        In practice, during training we step one fish at a time.
        """
        self.steps += 1
        
        # Move shark (sweeping pattern)
        self._move_shark()
        
        # Apply action to all alive fish
        total_reward = 0.0
        for i in range(self.num_fish):
            if self.fish_alive[i]:
                reward = self._move_fish(i, action)
                total_reward += reward
        
        # Check termination
        alive_count = np.sum(self.fish_alive)
        done = alive_count == 0 or self.steps >= self.max_steps
        
        # Get observation for next alive fish
        obs = self._get_observation(0)
        
        info = {
            "alive_fish": int(alive_count),
            "steps": self.steps,
            "eaten": int(self.num_fish - alive_count)
        }
        
        return obs, total_reward / max(1, self.num_fish), done, False, info
    
    def step_single_fish(self, fish_idx: int, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step a single fish - used for vectorized training.
        """
        if not self.fish_alive[fish_idx]:
            return self._get_observation(fish_idx), 0.0, True, {}
        
        reward = self._move_fish(fish_idx, action)
        obs = self._get_observation(fish_idx)
        done = not self.fish_alive[fish_idx]
        
        return obs, reward, done, {}
    
    def _move_shark(self):
        """
        Move shark in sweeping pattern: horizontal movement + upward drift.
        """
        # Horizontal movement
        self.shark_pos[0] += self.shark_speed * self.shark_direction * 3
        
        # Bounce off walls
        if self.shark_pos[0] >= self.width - self.shark_width / 2:
            self.shark_direction = -1
            self.shark_pos[1] += self.shark_height * 0.8  # Move up when bouncing
        elif self.shark_pos[0] <= self.shark_width / 2:
            self.shark_direction = 1
            self.shark_pos[1] += self.shark_height * 0.8
        
        # Reset to bottom if reached top
        if self.shark_pos[1] > self.height + self.shark_height:
            self.shark_pos[1] = -self.shark_height / 2
    
    def _move_fish(self, fish_idx: int, action: np.ndarray) -> float:
        """
        Move a single fish based on action, return reward.
        """
        pos = self.fish_positions[fish_idx]
        vel = self.fish_velocities[fish_idx]
        
        # Apply action as acceleration
        action = np.clip(action, -1, 1)
        vel += action * 0.5
        
        # Limit velocity
        speed = np.linalg.norm(vel)
        if speed > self.fish_speed:
            vel = vel / speed * self.fish_speed
        
        # Update position
        pos += vel
        
        # Bounce off walls
        if pos[0] < self.fish_radius:
            pos[0] = self.fish_radius
            vel[0] *= -0.5
        elif pos[0] > self.width - self.fish_radius:
            pos[0] = self.width - self.fish_radius
            vel[0] *= -0.5
        
        if pos[1] < self.fish_radius:
            pos[1] = self.fish_radius
            vel[1] *= -0.5
        elif pos[1] > self.height - self.fish_radius:
            pos[1] = self.height - self.fish_radius
            vel[1] *= -0.5
        
        # Check collision with shark
        reward = 0.1  # Survival reward
        
        # Distance to shark center
        dist_to_shark = np.linalg.norm(pos - self.shark_pos)
        
        # Collision check (ellipse approximation)
        dx = (pos[0] - self.shark_pos[0]) / (self.shark_width / 2 + self.fish_radius)
        dy = (pos[1] - self.shark_pos[1]) / (self.shark_height / 2 + self.fish_radius)
        
        if dx * dx + dy * dy < 1.0:
            # Fish got eaten!
            self.fish_alive[fish_idx] = False
            self.total_eaten += 1
            reward = -10.0
        else:
            # Reward for staying away from shark
            safe_distance = 100.0
            if dist_to_shark > safe_distance:
                reward += 0.05
            
            # Small penalty for being close to shark
            if dist_to_shark < safe_distance:
                reward -= 0.1 * (1 - dist_to_shark / safe_distance)
        
        self.fish_positions[fish_idx] = pos
        self.fish_velocities[fish_idx] = vel
        
        return reward
    
    def get_all_observations(self) -> List[np.ndarray]:
        """Get observations for all alive fish."""
        return [self._get_observation(i) for i in range(self.num_fish) if self.fish_alive[i]]
    
    def get_state_for_render(self) -> Dict:
        """Get full state for rendering."""
        return {
            "shark_pos": self.shark_pos.copy(),
            "shark_direction": self.shark_direction,
            "shark_size": (self.shark_width, self.shark_height),
            "fish_positions": self.fish_positions.copy(),
            "fish_velocities": self.fish_velocities.copy(),
            "fish_alive": self.fish_alive.copy(),
            "fish_radius": self.fish_radius,
            "width": self.width,
            "height": self.height,
            "steps": self.steps
        }
