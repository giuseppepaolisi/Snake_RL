
import torch
from env.agents import QLearningAgent
from env.snake_env import Snake_Env

import matplotlib.pyplot as plt
import os
import datetime

        
class Train:
    
    def __init__(self, env: Snake_Env, agent: QLearningAgent, episodes=1000, max_steps=200) -> None:
        self.episodes = episodes
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        
        
    def train(self):
        os.makedirs('models', exist_ok=True)
                
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            for step in range(self.max_steps):
                # L'agente sceglie un'azione
                action = self.agent.choose_action(state)

                # Ambiente esegue l'azione
                next_state, reward, done, info = self.env.step(action)
                
                # Impara dall'esperienza
                self.agent.update(state, action, reward, next_state, done)
                total_reward += reward

                # Passa al prossimo stato
                state = next_state

                if done:
                    break

            print(f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}, Score: {info['score']}")

        self.agent.save('models/snake_q_agent.pkl')
        print("Modello salvato in models/snake_q_agent.pkl")
        
        self.env.close()