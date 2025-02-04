import numpy as np
import matplotlib.pyplot as plt
import os
from agents import QLearningAgent, Sarsa

import numpy as np
import matplotlib.pyplot as plt
import os

class ComparisonPlotter:
    def __init__(self, models, episodes=10000, metrics_path='metrics'):
        """
        Inizializza il plotter per il confronto tra modelli.
        
        :param models: Lista dei nomi degli agenti da confrontare (es. ['Q-Learning', 'SARSA', 'DQN'])
        :param episodes: Numero di episodi del training
        :param metrics_path: Cartella dove sono salvate le metriche
        """
        self.models = models
        self.episodes = episodes
        self.metrics_path = metrics_path
        self.colors = ['blue', 'green', 'red', 'purple', 'orange']
        
    def _load_metric(self, metric_name):
        """Carica la metrica per tutti i modelli"""
        data = {}
        for model in self.models:
            path = os.path.join(self.metrics_path, f'{metric_name}_{model}_{self.episodes}.npy')
            if os.path.exists(path):
                data[model] = np.load(path)
            else:
                print(f"Attenzione: File non trovato per {model} - {metric_name}")
        return data
    
    def plot_rewards_comparison(self, window_size=100):
        """Grafico sovrapposto delle reward con media mobile"""
        rewards_data = self._load_metric('total_rewards')
        
        plt.figure(figsize=(14, 7))
        for i, (model, rewards) in enumerate(rewards_data.items()):
            if len(rewards) < window_size:
                continue
                
            # Calcola media mobile
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            
            # Plot
            plt.plot(range(window_size, len(rewards)+1), 
                    moving_avg, 
                    color=self.colors[i],
                    linewidth=2,
                    alpha=0.8,
                    label=f'{model}')
            
        plt.title(f'Confronto Reward ({self.episodes} episodi)', fontsize=14)
        plt.xlabel('Episodi', fontsize=12)
        plt.ylabel('Reward Media', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_path, f'rewards_comparison_{self.episodes}.png'))
        plt.show()
    
    def plot_scores_comparison(self, window_size=50):
        """Grafico sovrapposto degli score con smoothing"""
        scores_data = self._load_metric('score')
        
        plt.figure(figsize=(14, 7))
        for i, (model, scores) in enumerate(scores_data.items()):
            # Applica smoothing esponenziale
            smooth_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            
            plt.plot(range(window_size, len(scores)+1), 
                    smooth_scores, 
                    color=self.colors[i],
                    linewidth=1.5,
                    alpha=0.8,
                    label=f'{model}')
            
        plt.title('Confronto Score per Episodio', fontsize=14)
        plt.xlabel('Episodi', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_path, f'scores_comparison_{self.episodes}.png'))
        plt.show()
    
def main():
    # Configurazione
    models = ['Q-Learning', 'SARSA', 'DQN']
    episodes = 10000

    # Inizializzazione plotter
    plotter = ComparisonPlotter(
        models=models,
        episodes=episodes,
        metrics_path='metrics/'
    )

    # Generazione grafici
    plotter.plot_rewards_comparison(window_size=100)
    plotter.plot_scores_comparison(window_size=50)

if __name__ == "__main__":
    main()