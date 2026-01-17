"""
Predator-Prey AI Simulation
============================
A 2D simulation demonstrating Q-learning with predator and prey agents.

Run this file to train agents and generate a demo video.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from typing import List, Tuple
import time

from environment import Environment
from agents import Predator, Prey
from utils import get_state, RewardTracker, print_episode_stats


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Environment
    "grid_size": 20,

    # Training
    "training_episodes": 800,
    "max_steps_per_episode": 200,

    # Video recording
    "record_last_n_episodes": 5,  # Record these for the video
    "video_fps": 15,
    "video_filename": "demo.mp4",

    # Display
    "print_every": 100,
}


# ============================================================================
# Colors & Styling (Beautiful dark theme)
# ============================================================================

COLORS = {
    "background": "#0a0a0f",
    "grid_dark": "#12151a",
    "grid_light": "#1a1e25",
    "predator": "#ff4757",      # Vibrant red
    "predator_glow": "#ff6b7a",
    "prey": "#3498db",          # Electric blue
    "prey_glow": "#5dade2",
    "trail_predator": "#ff475755",
    "trail_prey": "#3498db55",
    "text": "#ecf0f1",
    "accent": "#00d4aa",
}


# ============================================================================
# Training Loop
# ============================================================================

def train(env: Environment, predator: Predator, prey: Prey,
          episodes: int, record_episodes: int = 5) -> Tuple[RewardTracker, List]:
    """
    Train both agents using Q-learning.
    Returns reward tracker and recorded frames from last episodes.
    """
    tracker = RewardTracker()
    recorded_episodes = []
    catches = 0

    print("\n" + "="*60)
    print("🎮 PREDATOR-PREY AI SIMULATION")
    print("="*60)
    print(f"Grid: {env.width}x{env.height} | Episodes: {episodes}")
    print("="*60 + "\n")

    for episode in range(episodes):
        pred_pos, prey_pos = env.reset()

        # Get initial states
        pred_state = get_state(env, is_predator=True)
        prey_state = get_state(env, is_predator=False)

        episode_pred_reward = 0
        episode_prey_reward = 0

        # For recording
        recording = episode >= (episodes - record_episodes)
        episode_frames = [] if recording else None

        done = False
        while not done:
            # Record frame
            if recording:
                episode_frames.append({
                    "predator": env.predator_pos,
                    "prey": env.prey_pos,
                    "step": env.steps
                })

            # Choose actions
            pred_action = predator.choose_action(pred_state)
            prey_action = prey.choose_action(prey_state)

            # Execute step
            pred_reward, prey_reward, done = env.step(pred_action, prey_action)

            # Get new states
            new_pred_state = get_state(env, is_predator=True)
            new_prey_state = get_state(env, is_predator=False)

            # Learn
            predator.learn(pred_state, pred_action, pred_reward, new_pred_state, done)
            prey.learn(prey_state, prey_action, prey_reward, new_prey_state, done)

            # Update states
            pred_state = new_pred_state
            prey_state = new_prey_state

            episode_pred_reward += pred_reward
            episode_prey_reward += prey_reward

        # Record final frame
        if recording:
            episode_frames.append({
                "predator": env.predator_pos,
                "prey": env.prey_pos,
                "step": env.steps,
                "final": True
            })
            recorded_episodes.append(episode_frames)

        # End of episode
        caught = env.predator_pos == env.prey_pos
        if caught:
            catches += 1

        predator.end_episode()
        prey.end_episode()
        predator.decay_epsilon()
        prey.decay_epsilon()

        tracker.add_episode(episode_pred_reward, episode_prey_reward,
                           env.steps, caught)

        # Progress output
        if (episode + 1) % CONFIG["print_every"] == 0:
            catch_rate = tracker.get_catch_rate()
            avg_len = tracker.get_avg_length()
            print(f"Episode {episode+1:4d} | "
                  f"Catch Rate: {catch_rate:.1%} | "
                  f"Avg Length: {avg_len:.1f} | "
                  f"ε: {predator.epsilon:.3f}")

    print("\n" + "="*60)
    print(f"✅ Training Complete! Total catches: {catches}/{episodes}")
    print(f"Final catch rate: {tracker.get_catch_rate():.1%}")
    print("="*60 + "\n")

    return tracker, recorded_episodes


# ============================================================================
# Visualization & Video Rendering
# ============================================================================

def create_video(env: Environment, recorded_episodes: List,
                 filename: str, fps: int = 15):
    """
    Create an animated video from recorded episodes.
    """
    print("🎬 Rendering video...")

    # Flatten all frames
    all_frames = []
    for ep_frames in recorded_episodes:
        all_frames.extend(ep_frames)
        # Add pause at episode end
        if ep_frames:
            for _ in range(fps // 2):  # Half second pause
                all_frames.append(ep_frames[-1])

    if not all_frames:
        print("No frames to render!")
        return

    # Setup figure with beautiful dark theme
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # Remove axes
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    title = ax.set_title("PREDATOR vs PREY", fontsize=24, fontweight='bold',
                         color=COLORS["text"], pad=20,
                         fontfamily='monospace')

    # Draw grid
    for i in range(env.height):
        for j in range(env.width):
            color = COLORS["grid_dark"] if (i + j) % 2 == 0 else COLORS["grid_light"]
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor=color, edgecolor='none')
            ax.add_patch(rect)

    # Create agent markers
    predator_circle = Circle((0, 0), 0.35, color=COLORS["predator"],
                             zorder=10, alpha=0.9)
    predator_glow = Circle((0, 0), 0.5, color=COLORS["predator_glow"],
                          zorder=9, alpha=0.3)
    prey_circle = Circle((0, 0), 0.3, color=COLORS["prey"],
                         zorder=10, alpha=0.9)
    prey_glow = Circle((0, 0), 0.45, color=COLORS["prey_glow"],
                      zorder=9, alpha=0.3)

    ax.add_patch(predator_circle)
    ax.add_patch(predator_glow)
    ax.add_patch(prey_circle)
    ax.add_patch(prey_glow)

    # Trail storage
    pred_trail = []
    prey_trail = []
    trail_patches = []

    # Stats text
    stats_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        fontsize=12, color=COLORS["text"],
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor=COLORS["grid_dark"],
                                 alpha=0.8, edgecolor=COLORS["accent"]))

    # Legend
    ax.text(0.98, 0.98, "🔴 Predator\n🔵 Prey", transform=ax.transAxes,
           fontsize=11, color=COLORS["text"],
           verticalalignment='top', horizontalalignment='right',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor=COLORS["grid_dark"],
                    alpha=0.8, edgecolor=COLORS["accent"]))

    def update(frame_idx):
        frame = all_frames[frame_idx]
        pred_pos = frame["predator"]
        prey_pos = frame["prey"]

        # Update positions
        predator_circle.center = pred_pos
        predator_glow.center = pred_pos
        prey_circle.center = prey_pos
        prey_glow.center = prey_pos

        # Update trails (keep last 8 positions)
        pred_trail.append(pred_pos)
        prey_trail.append(prey_pos)
        if len(pred_trail) > 8:
            pred_trail.pop(0)
        if len(prey_trail) > 8:
            prey_trail.pop(0)

        # Remove old trail patches
        for patch in trail_patches:
            patch.remove()
        trail_patches.clear()

        # Draw new trails with fading effect
        for i, pos in enumerate(pred_trail[:-1]):
            alpha = (i + 1) / len(pred_trail) * 0.4
            trail = Circle(pos, 0.15, color=COLORS["predator"], alpha=alpha, zorder=5)
            ax.add_patch(trail)
            trail_patches.append(trail)

        for i, pos in enumerate(prey_trail[:-1]):
            alpha = (i + 1) / len(prey_trail) * 0.4
            trail = Circle(pos, 0.12, color=COLORS["prey"], alpha=alpha, zorder=5)
            ax.add_patch(trail)
            trail_patches.append(trail)

        # Update stats
        dist = abs(pred_pos[0] - prey_pos[0]) + abs(pred_pos[1] - prey_pos[1])
        caught = "🎯 CAUGHT!" if pred_pos == prey_pos else ""
        stats_text.set_text(f"Step: {frame['step']:3d}\nDistance: {dist:2d}\n{caught}")

        # Pulse effect when caught
        if frame.get("final") and pred_pos == prey_pos:
            predator_glow.set_radius(0.7)
            predator_glow.set_alpha(0.6)
        else:
            predator_glow.set_radius(0.5)
            predator_glow.set_alpha(0.3)

        return [predator_circle, predator_glow, prey_circle, prey_glow,
                stats_text] + trail_patches

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(all_frames),
                                   interval=1000//fps, blit=False)

    # Save video
    print(f"Saving to {filename}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                    extra_args=['-vcodec', 'libx264'])
    try:
        anim.save(filename, writer=writer, dpi=100)
        print(f"✅ Video saved: {filename}")
    except Exception as e:
        print(f"FFmpeg not available, trying pillow...")
        try:
            writer = animation.PillowWriter(fps=fps)
            gif_filename = filename.replace('.mp4', '.gif')
            anim.save(gif_filename, writer=writer, dpi=80)
            print(f"✅ GIF saved: {gif_filename}")
        except Exception as e2:
            print(f"❌ Could not save video: {e2}")
            print("Install ffmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")

    plt.close(fig)


def plot_learning_curves(tracker: RewardTracker, save_path: str = "learning_curves.png"):
    """
    Plot learning curves showing agent improvement over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=COLORS["background"])
    fig.suptitle("Learning Curves", fontsize=18, color=COLORS["text"], fontweight='bold')

    for ax in axes.flat:
        ax.set_facecolor(COLORS["grid_dark"])
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["accent"])

    # Smooth data
    window = 50

    # Plot 1: Catch rate over time
    if tracker.catches:
        catch_rate = []
        for i in range(len(tracker.catches)):
            start = max(0, i - window)
            catch_rate.append(sum(tracker.catches[start:i+1]) / (i - start + 1))
        axes[0, 0].plot(catch_rate, color=COLORS["predator"], linewidth=2)
        axes[0, 0].set_title("Catch Rate", color=COLORS["text"])
        axes[0, 0].set_xlabel("Episode", color=COLORS["text"])
        axes[0, 0].set_ylabel("Rate", color=COLORS["text"])
        axes[0, 0].fill_between(range(len(catch_rate)), catch_rate,
                                alpha=0.3, color=COLORS["predator"])

    # Plot 2: Episode length
    if tracker.episode_lengths:
        axes[0, 1].plot(tracker.episode_lengths, color=COLORS["accent"],
                       alpha=0.3, linewidth=1)
        # Moving average
        if len(tracker.episode_lengths) > window:
            ma = np.convolve(tracker.episode_lengths,
                           np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(tracker.episode_lengths)), ma,
                          color=COLORS["accent"], linewidth=2)
        axes[0, 1].set_title("Episode Length", color=COLORS["text"])
        axes[0, 1].set_xlabel("Episode", color=COLORS["text"])
        axes[0, 1].set_ylabel("Steps", color=COLORS["text"])

    # Plot 3: Predator rewards
    if tracker.predator_rewards:
        axes[1, 0].plot(tracker.predator_rewards, color=COLORS["predator"],
                       alpha=0.3, linewidth=1)
        if len(tracker.predator_rewards) > window:
            ma = np.convolve(tracker.predator_rewards,
                           np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(tracker.predator_rewards)), ma,
                          color=COLORS["predator"], linewidth=2)
        axes[1, 0].set_title("Predator Reward", color=COLORS["text"])
        axes[1, 0].set_xlabel("Episode", color=COLORS["text"])
        axes[1, 0].set_ylabel("Reward", color=COLORS["text"])

    # Plot 4: Prey rewards
    if tracker.prey_rewards:
        axes[1, 1].plot(tracker.prey_rewards, color=COLORS["prey"],
                       alpha=0.3, linewidth=1)
        if len(tracker.prey_rewards) > window:
            ma = np.convolve(tracker.prey_rewards,
                           np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(tracker.prey_rewards)), ma,
                          color=COLORS["prey"], linewidth=2)
        axes[1, 1].set_title("Prey Reward", color=COLORS["text"])
        axes[1, 1].set_xlabel("Episode", color=COLORS["text"])
        axes[1, 1].set_ylabel("Reward", color=COLORS["text"])

    plt.tight_layout()
    plt.savefig(save_path, facecolor=COLORS["background"], dpi=150,
               bbox_inches='tight')
    print(f"✅ Learning curves saved: {save_path}")
    plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the complete simulation pipeline."""

    # Initialize
    env = Environment(width=CONFIG["grid_size"], height=CONFIG["grid_size"])
    predator = Predator(
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    prey = Prey(
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05  # Prey keeps some randomness
    )

    # Train
    tracker, recorded_episodes = train(
        env, predator, prey,
        episodes=CONFIG["training_episodes"],
        record_episodes=CONFIG["record_last_n_episodes"]
    )

    # Generate outputs
    create_video(env, recorded_episodes, CONFIG["video_filename"], CONFIG["video_fps"])
    plot_learning_curves(tracker)

    # Save trained agents
    predator.save("predator_model.pkl")
    prey.save("prey_model.pkl")
    print("✅ Models saved!")

    print("\n" + "="*60)
    print("🎉 ALL DONE!")
    print("="*60)
    print(f"📹 Video: {CONFIG['video_filename']}")
    print("📊 Charts: learning_curves.png")
    print("🤖 Models: predator_model.pkl, prey_model.pkl")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
