# Snake Reinforcement Learning

A Python implementation of the classic Snake game with Reinforcement Learning agents. This project uses OpenAI Gym (formerly OpenAI Gym) to create a custom Snake environment where different RL agents can be trained to play the game.

Supported agents: DQN, Sarsa, QLearning

<div align="center">
  <img src="immage\DQN_game.gif" width="300" alt="Demo">
  <p>Agent DQN</p>
</div>

## 🚀 Features

- Custom Snake environment built with OpenAI Gym
- Multiple RL agent implementations
- Configurable reward system
- Performance metrics tracking
- Visualization of training progress
- Trained model saving and loading capabilities

## 📊 Performance Metrics


### Reward Graph

<div align="center">
  <img src="immage\reward.png" width="500" alt="Demo">
  <p>Plots the average reward over time</p>
</div>

### Score Graph

<div align="center">
    <img src="immage\score.png" width="500" alt="Demo">
    <p>Plots the average score over time</p>
</div>

## 🛠️ Requirements

- Python 3.12+
- PyGame
- gym
- NumPy
- Matplotlib
- Pandas
- PyTorch

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/giuseppepaolisi/Snake_RL.git
cd Snake_RL
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

1. To train a new agent:
```bash
python src/main.py
```

## 📁 Project Structure

```
Snake_RL/
├── src/
│   ├── agents.py           # RL agent implementations
│   ├── main.py            # Main training script
│   └── env/
│       ├── snake_env.py   # Snake environment
│       └── snake/
│           └── const/
│               ├── actions.py  # Action space definitions
│               └── rewards.py  # Reward system configuration
├── models/                 # Saved model weights
├── metrics/               # Training metrics and logs
└── requirements.txt       # Project dependencies
```

##
Copyright © 2024
[Giuseppe Paolisi](https://github.com/giuseppepaolisi)
[Cristian Fulchini](https://github.com/cris83040).