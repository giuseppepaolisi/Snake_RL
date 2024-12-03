
import torch
from env.agents import QLearningAgent
from env.snake_env import Snake_Env

import matplotlib.pyplot as plt
import os
import datetime

class TrainingLogger:
    def __init__(self, save_dir='training_logs'):
        self.rewards = []
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def log_episode(self, reward):
        self.rewards.append(reward)

    def plot_training_metrics(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f'training_rewards_{timestamp}.png')
        plt.savefig(filename)
        plt.close()
        print(f"Grafico salvato in {filename}")
        
class Train:
    
    def __init__(self, env: Snake_Env, agent: QLearningAgent, episodes=1000, max_steps=200) -> None:
        self.episodes = episodes
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        
        
        #self.logger = TrainingLogger()
    
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
                
            # Salvataggio ricompensa totale per episodio
            #self.logger.log_episode(total_reward)

            print(f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}, Score: {info['score']}")

        self.agent.save('models/snake_q_agent.pkl')
        print("Modello salvato in models/snake_q_agent.pkl")
        
        # Salva il grafico a fine training
        #self.logger.plot_training_metrics()
        
        self.env.close()