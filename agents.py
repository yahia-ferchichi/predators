"""
Q-Learning Agents for Predator-Prey Simulation
"""

import numpy as np
from typing import Tuple, Dict
import pickle


class QLearningAgent:
    """
    Tabular Q-Learning Agent.

    State space: (relative_x, relative_y, distance_bucket)
    Action space: 0=up, 1=down, 2=left, 3=right, 4=stay
    """

    def __init__(
        self,
        name: str,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.name = name
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> action values
        # State: (rel_x, rel_y, dist) where rel_x, rel_y in {0,1,2} and dist in {0,1,2,3}
        # Actions: 5 possible moves
        self.q_table: Dict[Tuple, np.ndarray] = {}

        self.n_actions = 5

        # Statistics
        self.total_reward = 0.0
        self.episode_rewards = []

    def _get_q_values(self, state: Tuple) -> np.ndarray:
        """Get Q-values for a state, initialize if new."""
        if state not in self.q_table:
            # Initialize with small random values for exploration
            self.q_table[state] = np.random.uniform(-0.1, 0.1, self.n_actions)
        return self.q_table[state]

    def choose_action(self, state: Tuple, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best known action
            q_values = self._get_q_values(state)
            return int(np.argmax(q_values))

    def learn(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool
    ):
        """
        Update Q-table using Q-learning update rule.
        Q(s,a) = Q(s,a) + lr * (reward + gamma * max(Q(s',a')) - Q(s,a))
        """
        current_q = self._get_q_values(state)[action]

        if done:
            target = reward
        else:
            next_q_values = self._get_q_values(next_state)
            target = reward + self.gamma * np.max(next_q_values)

        # Q-learning update
        self.q_table[state][action] += self.lr * (target - current_q)

        self.total_reward += reward

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def end_episode(self):
        """Record episode statistics and reset."""
        self.episode_rewards.append(self.total_reward)
        self.total_reward = 0.0

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
        return {
            "name": self.name,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "avg_reward_last_100": np.mean(recent_rewards),
            "total_episodes": len(self.episode_rewards)
        }

    def save(self, filepath: str):
        """Save agent to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards
            }, f)

    def load(self, filepath: str):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            self.episode_rewards = data['episode_rewards']


class Predator(QLearningAgent):
    """
    Predator agent - learns to chase and catch prey.
    """

    def __init__(self, **kwargs):
        super().__init__(name="Predator", **kwargs)
        # Predator can be slightly faster in learning
        self.lr = kwargs.get('learning_rate', 0.15)


class Prey(QLearningAgent):
    """
    Prey agent - learns to escape from predator.
    """

    def __init__(self, **kwargs):
        super().__init__(name="Prey", **kwargs)
        # Prey might need to learn faster to survive
        self.lr = kwargs.get('learning_rate', 0.2)
