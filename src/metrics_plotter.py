import numpy as np
import matplotlib.pyplot as plt

def plot_combined_metric(metric, episode, agents):
    """
    Plots the combined metric (e.g., score or reward) for multiple agents.

    Args:
        metric (str): The metric to plot (e.g., 'score', 'reward').
        episodes (int): The number of episodes for which data is available.
        agents (list): A list of tuples, each containing an agent and its label.
                       Example: [(sarsa_agent, 'Sarsa'), (q_learning_agent, 'Q-Learning')]
    """
    plt.figure(figsize=(12, 6))

    for agent, label in agents:
        # Percorso del file delle metriche
        rewards_path = f'metrics/{metric}_{agent.get_model()}_{episode}.npy'
        values = np.load(rewards_path)
        
        # Linea continua per ogni agente
        plt.plot(values, linewidth=1.5, label=label)

    plt.xlabel("Episodes")
    plt.ylabel(metric.capitalize())
    plt.title(f"Comparison of {metric.capitalize()} Across Agents")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)  # Aggiunge una griglia leggera
    plt.tight_layout()  # Ottimizza il layout
    plt.savefig(f'metrics/combined_{metric}_{episode}.png')
    plt.show()
