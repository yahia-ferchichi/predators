"""
Train Fish with Deep RL (PPO)
=============================
Train fish to escape the shark using Proximal Policy Optimization.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import List, Tuple, Dict
import os

from environment import OceanEnvironment
from visualization import OceanRenderer, VideoRecorder


class PolicyNetwork(nn.Module):
    """
    Shared policy network for all fish.
    Input: raycast observations
    Output: mean and std for action distribution
    """
    
    def __init__(self, obs_dim: int, action_dim: int = 2, hidden_size: int = 128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        action_mean, value = self.forward(obs)
        
        if deterministic:
            return action_mean, value, None
        
        std = torch.exp(self.actor_logstd)
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        
        return action, value, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        action_mean, value = self.forward(obs)
        std = torch.exp(self.actor_logstd)
        dist = Normal(action_mean, std)
        
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return value, log_prob, entropy


class PPOTrainer:
    """
    PPO trainer for the fish escape task.
    """
    
    def __init__(
        self,
        env: OceanEnvironment,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "cpu"
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        
        # Initialize policy
        self.policy = PolicyNetwork(env.obs_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Rollout storage
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []
        self.done_buffer = []
    
    def collect_rollout(self, n_steps: int = 2048, render: bool = False, 
                       renderer: OceanRenderer = None) -> Dict:
        """Collect experience from environment."""
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.value_buffer.clear()
        self.log_prob_buffer.clear()
        self.done_buffer.clear()
        
        obs, _ = self.env.reset()
        total_reward = 0
        episode_rewards = []
        current_episode_reward = 0
        
        for step in range(n_steps):
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, value, log_prob = self.policy.get_action(obs_tensor)
            
            action_np = action.cpu().numpy().flatten()
            
            # Store transition
            self.obs_buffer.append(obs)
            self.action_buffer.append(action_np)
            self.value_buffer.append(value.item())
            self.log_prob_buffer.append(log_prob.item())
            
            # Step environment (apply same action to all fish with slight variation)
            for i in range(self.env.num_fish):
                if self.env.fish_alive[i]:
                    # Add small random noise for diversity
                    fish_action = action_np + np.random.normal(0, 0.1, size=action_np.shape)
                    obs_i, reward_i, done_i, _ = self.env.step_single_fish(i, fish_action)
            
            # Move shark
            self.env._move_shark()
            self.env.steps += 1
            
            # Calculate collective reward
            alive_reward = np.sum(self.env.fish_alive) * 0.01
            reward = alive_reward
            
            self.reward_buffer.append(reward)
            current_episode_reward += reward
            total_reward += reward
            
            # Check if episode done
            done = np.sum(self.env.fish_alive) == 0 or self.env.steps >= self.env.max_steps
            self.done_buffer.append(done)
            
            # Render if requested
            if render and renderer:
                state = self.env.get_state_for_render()
                if not renderer.render(state, training=True):
                    break
            
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                obs, _ = self.env.reset()
            else:
                # Get next observation from first alive fish
                for i in range(self.env.num_fish):
                    if self.env.fish_alive[i]:
                        obs = self.env._get_observation(i)
                        break
        
        return {
            "total_reward": total_reward,
            "episode_rewards": episode_rewards,
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0
        }
    
    def compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.reward_buffer)
        values = np.array(self.value_buffer)
        dones = np.array(self.done_buffer)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self) -> Dict:
        """Update policy using PPO."""
        advantages, returns = self.compute_gae()
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        action_tensor = torch.FloatTensor(np.array(self.action_buffer)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_prob_buffer)).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # PPO update for multiple epochs
        total_loss = 0
        policy_losses = []
        value_losses = []
        
        n_samples = len(self.obs_buffer)
        indices = np.arange(n_samples)
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # Get batch
                batch_obs = obs_tensor[batch_idx]
                batch_actions = action_tensor[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                
                # Evaluate actions
                values, log_probs, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
                values = values.squeeze()
                
                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        return {
            "total_loss": total_loss / self.n_epochs,
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses)
        }
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
        print(f"✅ Model saved: {path}")
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"✅ Model loaded: {path}")


def train(
    n_iterations: int = 100,
    steps_per_iteration: int = 2048,
    render_every: int = 10,
    save_path: str = "fish_policy.pth"
):
    """
    Main training loop.
    """
    print("\n" + "=" * 60)
    print("🧠 TRAINING FISH WITH PPO")
    print("=" * 60)
    
    # Initialize
    env = OceanEnvironment(
        width=800,
        height=600,
        num_fish=15,
        num_rays=24,
        shark_speed=2.0,
        fish_speed=4.0
    )
    
    trainer = PPOTrainer(env)
    renderer = None
    
    best_reward = -float('inf')
    
    for iteration in range(n_iterations):
        # Render occasionally
        render = (iteration + 1) % render_every == 0
        if render and renderer is None:
            renderer = OceanRenderer(env.width, env.height)
        
        # Collect rollout
        rollout_info = trainer.collect_rollout(
            n_steps=steps_per_iteration,
            render=render,
            renderer=renderer
        )
        
        # Update policy
        update_info = trainer.update_policy()
        
        # Logging
        mean_reward = rollout_info["mean_reward"]
        print(f"Iteration {iteration + 1:3d} | "
              f"Mean Reward: {mean_reward:8.2f} | "
              f"Policy Loss: {update_info['policy_loss']:.4f} | "
              f"Value Loss: {update_info['value_loss']:.4f}")
        
        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            trainer.save(save_path)
    
    if renderer:
        renderer.close()
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Model saved: {save_path}")
    print("=" * 60)
    
    return trainer


if __name__ == "__main__":
    train(n_iterations=50, render_every=10)
