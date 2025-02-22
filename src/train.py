from env.snake_env import Snake_Env
import matplotlib.pyplot as plt
import os
import json
import numpy as np

class Train:
    
    def __init__(self, env: Snake_Env, agent, episodes=1000, max_steps=200) -> None:
        self.episodes = episodes
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        
        # Crea directory per modelli e metriche
        os.makedirs('models', exist_ok=True)
        os.makedirs('metrics', exist_ok=True)
        
  
    def train(self):
        model_name = self.agent.get_model()
        total_rewards = []
        eps = []
        score = []
        for episode in range(self.episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            
            while True:
            #for step in range(self.max_steps):
                # L'agente sceglie un'azione
                action = self.agent.choose_action(state)

                # Ambiente esegue l'azione
                next_state, reward, done, _, info = self.env.step(action)
                                
                self.agent.update(state, action, reward, next_state, done)
                
                total_reward += reward

                # Passa al prossimo stato
                state = next_state
                
                if done:
                    break
            
            self.agent.decay_epsilon(episodes_completed=(episode + 1))       
            print(f'Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}, Score: {info["score"]}')
            total_rewards.append(total_reward)
            eps.append(self.agent.epsilon)
            score.append(info["score"])
            
        model_path = f'models/snake_{model_name}_{self.episodes}.pkl'
        self.agent.save(model_path)
        print(f'Modello salvato in {model_path}')
        
        # Salvataggio ricompense totali
        rewards_path = f'metrics/total_rewards_{model_name}_{self.agent.decay_mode}_{self.episodes}.npy'
        np.save(rewards_path, np.array(total_rewards))
        print(f'Ricompense totali salvate in {rewards_path}')
        
        # Salvataggio decremento epsilon
        rewards_path = f'metrics/epsilon_decay_{model_name}_{self.agent.decay_mode}_{self.episodes}.npy'
        np.save(rewards_path, np.array(eps))
        print(f'Ricompense totali salvate in {rewards_path}')
        
        # Salvataggio score
        rewards_path = f'metrics/score_{model_name}_{self.agent.decay_mode}_{self.episodes}.npy'
        np.save(rewards_path, np.array(score))
        print(f'Ricompense totali salvate in {rewards_path}')

        # Grafico delle metriche
        self.plot_metrics()
        self.plot_eps()
        self.plot_score()
                
        self.env.close()
    
    def plot_metrics(self):
        """Genera grafici delle metriche raccolte."""
        # Carica i dati salvati
        rewards_path = f'metrics/total_rewards_{self.agent.get_model()}_{self.agent.decay_mode}_{self.episodes}.npy'
        rewards = np.load(rewards_path)

        # Calcola la media mobile
        window_size = 100
        moving_avg = np.convolve(rewards, np.ones((window_size,)) / window_size, mode="valid")

        # Grafico
        plt.figure(figsize=(12, 6))
        plt.plot(range(window_size, len(rewards) + 1), moving_avg, label="Rewards Average", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f" Rewards ({self.agent.get_model()})")
        plt.legend()
        plt.savefig(f'metrics/rewards_{self.agent.get_model()}_{self.episodes}.png')
        plt.show()
    
    def plot_eps(self):
        # Carica i dati salvati
        rewards_path = f'metrics/epsilon_decay_{self.agent.get_model()}_{self.agent.decay_mode}_{self.episodes}.npy'
        eps = np.load(rewards_path)

        # Grafico
        plt.figure(figsize=(12, 6))
        plt.plot(eps, label="Epsilon", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title(f"Epsilon decay ({self.agent.get_model()})")
        plt.legend()
        plt.savefig(f'metrics/epsilon_decay_{self.agent.get_model()}_{self.agent.decay_mode}_{self.episodes}.png')
        plt.show()
        
    def plot_score(self):
        # Carica i dati salvati
        rewards_path = f'metrics/score_{self.agent.get_model()}_{self.agent.decay_mode}_{self.episodes}.npy'
        score = np.load(rewards_path)
        
        # Grafico
        plt.figure(figsize=(12, 6))
        plt.plot(score, label="Score", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title(f"Score ({self.agent.get_model()})")
        plt.legend()
        plt.savefig(f'metrics/score_{self.agent.get_model()}_{self.agent.decay_mode}_{self.episodes}.png')
        plt.show()