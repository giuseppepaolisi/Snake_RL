
import torch
from env.agents import QLearningAgent
from env.snake_env import Snake_Env

import matplotlib.pyplot as plt
import os
import json
import numpy as np

        
class Train:
    
    def __init__(self, env: Snake_Env, agent: QLearningAgent, episodes=1000, max_steps=200) -> None:
        self.episodes = episodes
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        
        # Crea directory per modelli e metriche
        os.makedirs('models', exist_ok=True)
        os.makedirs('metrics', exist_ok=True)
        
        
    def train(self):
        total_rewards = []
        for episode in range(self.episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            
            while True:
            #for step in range(self.max_steps):
                # L'agente sceglie un'azione
                action = self.agent.choose_action(state)

                # Ambiente esegue l'azione
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Impara dall'esperienza
                self.agent.update(state, action, reward, next_state, done)
                total_reward += reward

                # Passa al prossimo stato
                state = next_state

                #if reward > 0:
                #    step = 0
                
                if done:
                    break
                   
            print(f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}, Score: {info['score']}")
            total_rewards.append(total_reward)

        model_path = f'models/snake_q_agent_{self.episodes}.pkl'
        self.agent.save(model_path)
        print(f"Modello salvato in {model_path}")
        
        # Salvataggio ricompense totali
        rewards_path = f'metrics/total_rewards_{self.episodes}.npy'
        np.save(rewards_path, np.array(total_rewards))
        print(f"Ricompense totali salvate in {rewards_path}")

        # Grafico delle metriche
        self.plot_metrics()
        
        self.env.close()
    
    def plot_metrics(self):
        """Genera grafici delle metriche raccolte."""
        # Carica i dati salvati
        rewards_path = f'metrics/total_rewards_{self.episodes}.npy'
        rewards = np.load(rewards_path)

        # Calcola la media mobile
        window_size = 100
        moving_avg = np.convolve(rewards, np.ones((window_size,)) / window_size, mode="valid")

        # Grafico
        plt.figure(figsize=(12, 6))
        plt.plot(range(window_size, len(rewards) + 1), moving_avg, label="Moving Average", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Moving Average of Rewards")
        plt.legend()
        plt.savefig(f'metrics/moving_average_rewards_{self.episodes}.png')
        plt.show()
    