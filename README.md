# 🎮 Predator-Prey AI Simulation

A 2D simulation demonstrating emergent behavior through reinforcement learning. Watch as AI agents learn to hunt and survive!

![Demo](demo.gif)

## 🧠 Overview

This project implements a predator-prey ecosystem where:

- **Predators** 🔴 learn to chase and catch prey using Q-learning
- **Prey** 🔵 learn survival strategies to escape predators
- Both agents start with random behavior and develop intelligent strategies over time

## ✨ Features

- **Tabular Q-Learning**: Simple yet effective RL algorithm
- **Real-time Learning**: Watch agents improve episode by episode
- **Beautiful Visualization**: Dark-themed animation with trails and effects
- **Learning Curves**: Track performance metrics over training
- **Model Persistence**: Save and load trained agents

## 🛠️ Tech Stack

- Python 3.10+
- NumPy - numerical operations
- Matplotlib - visualization & video rendering

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-sim-predator-prey.git
cd ai-sim-predator-prey

# Install dependencies
pip install -r requirements.txt

# Optional: Install FFmpeg for MP4 export
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

### Run Simulation

```bash
python main.py
```

This will:
1. Train both agents for 800 episodes
2. Generate `demo.mp4` (or `demo.gif` if FFmpeg unavailable)
3. Save learning curves to `learning_curves.png`
4. Export trained models as `.pkl` files

## 📊 How It Works

### Environment
- 20×20 grid world
- Agents can move: ↑ ↓ ← → or stay
- Episode ends when predator catches prey or timeout (200 steps)

### State Space
Each agent observes:
- Relative direction to opponent (discretized)
- Distance category (close/medium/far)

### Reward Structure

| Agent | Reward |
|-------|--------|
| Predator | +10 catch, +0.5 per cell closer, -0.1 step penalty |
| Prey | -10 caught, +0.5 per cell farther, +0.1 survival bonus |

### Learning Parameters
- Learning rate (α): 0.15 (predator), 0.2 (prey)
- Discount factor (γ): 0.95
- Exploration: ε-greedy with decay (1.0 → 0.01)

## 📁 Project Structure

```
ai-sim-predator-prey/
├── main.py           # Training loop & visualization
├── agents.py         # Q-learning agent classes
├── environment.py    # 2D grid environment
├── utils.py          # Helper functions
├── requirements.txt  # Dependencies
├── README.md
├── demo.mp4          # Generated video
└── learning_curves.png
```

## 📈 Results

After training, you should see:
- **Catch rate**: ~60-80% (varies by run)
- **Learned behaviors**:
  - Predator: Direct pursuit, cutting off escape routes
  - Prey: Evasive movement, boundary awareness

## 🎬 Demo Video

The generated video shows trained agents in action:
- Red dot = Predator
- Blue dot = Prey
- Fading trails show movement history
- Stats display step count and distance

## 🔧 Configuration

Edit `CONFIG` in `main.py` to customize:

```python
CONFIG = {
    "grid_size": 20,              # World size
    "training_episodes": 800,     # Training duration
    "video_fps": 15,              # Output video FPS
    "record_last_n_episodes": 5,  # Episodes to record
}
```

## 📝 License

MIT License - feel free to use for your own projects!

## 🙏 Acknowledgments

Inspired by classic predator-prey models in ecology and multi-agent reinforcement learning research.

---

*Built with 🔥 for learning and portfolio demonstration*
