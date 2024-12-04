from gym.envs.registration import register
import gym
from env.agents import QLearningAgent
import numpy as np
from train import Train

register(
    id='Snake-v0',
    entry_point='env.new_snake_env:Snake_Env',
    max_episode_steps=300,
)
env = gym.make('Snake-v0', render_mode='human')
num_episodes = 10
for i in range(num_episodes):
    done = False
    env.reset()
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, th, info = env.step(action)
    