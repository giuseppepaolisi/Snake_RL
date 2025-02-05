from env.snake_env import Snake_Env
from agents import QLearningAgent, Sarsa, DQN
import matplotlib.pyplot as plt
import numpy as np
import os

def test_agent(agent, num_episodes=100, size=5):
    env = Snake_Env(size=size)
    
    # Directory per salvare i grafici
    os.makedirs('metrics/test', exist_ok=True)
    
    total_rewards = []
    scores = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        print(f'--- Episode {episode + 1} ---')

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)  # Ignoriamo truncated
            total_reward += reward
            state = next_state
            
            if done:
                break

        print(f'Total reward for episode {episode + 1}: {total_reward} Score: {info["score"]}')
        total_rewards.append(total_reward)
        scores.append(info["score"])
        
    env.close()
    
    # Generazione dei grafici con distinzione di modello, metrica e numero di episodi
    plot_metrics(agent, total_rewards, "rewards", num_episodes)
    plot_metrics(agent, scores, "scores", num_episodes)

    
def plot_metrics(agent, data, metric, num_episodes):
    """Genera grafici delle metriche raccolte durante il test."""
    # Salva i dati in un file `.npy` per eventuale analisi futura
    data_path = f'metrics/test/total_{metric}_{agent.get_model()}_{num_episodes}.npy'
    np.save(data_path, data)
    
    # Grafico
    # Calcola la media mobile
    window_size = 100
    moving_avg = np.convolve(data, np.ones((window_size,)) / window_size, mode="valid")
    plt.figure(figsize=(12, 6))
    plt.plot(range(window_size, len(data) + 1), moving_avg, label=f"Total {metric.capitalize()}", color='blue', alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} ({agent.get_model()})")
    plt.legend()
    
    # Salvataggio del grafico
    plot_path = f'metrics/test/{metric}_{agent.get_model()}_{num_episodes}.png'
    plt.savefig(plot_path)
    plt.show()
