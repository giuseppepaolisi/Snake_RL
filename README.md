# Snake Reinforcement Learning

A Python implementation of the classic Snake game with Reinforcement Learning agents. This project uses OpenAI Gym (formerly OpenAI Gym) to create a custom Snake environment where different RL agents can be trained to play the game.

## 🎮 Project Overview

This project implements a Snake game environment and trains various reinforcement learning agents to play it. The agents learn to control the snake, collect food, and avoid collisions with walls and themselves.

(Add screenshots or GIFs of the game in action)

## 🚀 Features

- Custom Snake environment built with OpenAI Gym
- Multiple RL agent implementations
- Configurable reward system
- Performance metrics tracking
- Visualization of training progress
- Trained model saving and loading capabilities

## 🛠️ Requirements

- Python 3.12+
- PyGame
- Gymnasium
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

2. To test a trained agent:
```bash
python tester.py
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

## 📊 Performance Metrics

Training metrics are saved in the `metrics` directory. You can visualize the training progress using the included plotting utilities.

(Add graphs or charts of training metrics)

##
Copyright © 2024
[Giuseppe Paolisi](https://github.com/giuseppepaolisi)
[Cristian Fulchini](https://github.com/cris83040).