import numpy as np
import matplotlib.pyplot as plt
import os

class ComparisonPlotter:
    def __init__(self, models, episodes=10000, epsilon_mode='exponential', metrics_path='metrics'):
        """
        Inizializza il plotter per il confronto tra modelli.
        
        :param models: Lista dei nomi degli agenti da confrontare (es. ['Q-Learning', 'SARSA', 'DQN'])
        :param episodes: Numero di episodi del training
        :param metrics_path: Cartella dove sono salvate le metriche
        """
        self.models = models
        self.episodes = episodes
        self.metrics_path = metrics_path
        self.epsilon_mode = epsilon_mode
        self.colors = ['blue', 'green', 'red', 'purple', 'orange']
        # Lista degli epsilon mode da confrontare
        self.epsilon_modes = ['exponential', 'linear', 'cosine', 'step']
        
    def _load_metric(self, metric_name):
        """Carica la metrica per tutti i modelli"""
        data = {}
        for model in self.models:
            path = os.path.join(self.metrics_path, f'{metric_name}_{model}_{self.epsilon_mode}_{self.episodes}.npy')
            if os.path.exists(path):
                data[model] = np.load(path)
            else:
                print(f"Attenzione: File non trovato per {model} - {metric_name}")
        return data
    
    def _load_metrics(self, metric_names, model):
        """Carica le metriche per un modello e tutte le epsilon mode (exponential, linear, cosine, step)"""
        data = {}
        for epsilon_mode in self.epsilon_modes:
            path = os.path.join(self.metrics_path, f'{metric_names}_{model}_{epsilon_mode}_{self.episodes}.npy')
            if os.path.exists(path):
                data[epsilon_mode] = np.load(path)
            else:
                print(f"Attenzione: File non trovato per {model} - {metric_names} - {epsilon_mode}")
        return data
    
    def plot_rewards_comparison(self, window_size=100):
        """Grafico sovrapposto delle reward con media mobile per tutti i modelli"""
        rewards_data = self._load_metric('total_rewards')
        
        plt.figure(figsize=(14, 7))
        for i, (model, rewards) in enumerate(rewards_data.items()):
            if len(rewards) < window_size:
                continue
                
            # Calcola media mobile
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            
            # Plot della moving average
            plt.plot(range(window_size, len(rewards)+1), 
                     moving_avg, 
                     color=self.colors[i % len(self.colors)],
                     linewidth=2,
                     alpha=0.8,
                     label=f'{model} (media mobile)')
            
            # Plot della media reward per episodio
            mean_reward = np.mean(rewards)
            plt.plot([0, len(rewards)], [mean_reward, mean_reward], 
                     color=self.colors[i % len(self.colors)], 
                     linestyle='--', 
                     label=f'{model} media')
            
        plt.title(f'Confronto Reward ({self.episodes} episodi)', fontsize=14)
        plt.xlabel('Episodi', fontsize=12)
        plt.ylabel('Reward Media', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_path, f'rewards_comparison_{self.episodes}.png'))
        plt.show()
    
    def plot_scores_comparison(self, window_size=50):
        """Grafico sovrapposto degli score con smoothing per tutti i modelli"""
        scores_data = self._load_metric('score')
        
        plt.figure(figsize=(14, 7))
        for i, (model, scores) in enumerate(scores_data.items()):
            if len(scores) < window_size:
                continue
                
            # Applica smoothing con una media mobile
            smooth_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            
            plt.plot(range(window_size, len(scores)+1), 
                     smooth_scores, 
                     color=self.colors[i % len(self.colors)],
                     linewidth=1.5,
                     alpha=0.8,
                     label=f'{model} (smooth)')
            
            # Plot della media score per episodio
            mean_score = np.mean(scores)
            plt.plot([0, len(scores)], [mean_score, mean_score], 
                     color=self.colors[i % len(self.colors)], 
                     linestyle='--', 
                     label=f'{model} media')
            
        plt.title('Confronto Score per Episodio', fontsize=14)
        plt.xlabel('Episodi', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_path, f'scores_comparison_{self.episodes}.png'))
        plt.show()
    
    def plot_rewards_epsilon_comparison(self, model, window_size=100):
        """Grafico delle reward in funzione di epsilon per un modello.
        
        Per il modello specificato, carica le reward per ciascun epsilon mode e le confronta.
        """
        # Carica le reward per tutte le epsilon mode per il modello specificato
        rewards_data = self._load_metrics('total_rewards', model)
        
        plt.figure(figsize=(14, 7))
        for i, epsilon_mode in enumerate(self.epsilon_modes):
            if epsilon_mode not in rewards_data:
                continue
            
            rewards = rewards_data[epsilon_mode]
            if len(rewards) < window_size:
                continue
            
            # Calcola la moving average
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            
            # Plot della moving average
            plt.plot(range(window_size, len(rewards)+1),
                     moving_avg,
                     color=self.colors[i % len(self.colors)],
                     linewidth=2,
                     alpha=0.8,
                     label=f'{model} - {epsilon_mode}')
            
            # Plot della media reward per episodio
            mean_reward = np.mean(rewards)
            plt.plot([0, len(rewards)], [mean_reward, mean_reward],
                     color=self.colors[i % len(self.colors)],
                     linestyle='--',
                     label=f'{model} {epsilon_mode} media')
        
        plt.title(f'Confronto Reward per Epsilon Mode ({model})', fontsize=14)
        plt.xlabel('Episodi', fontsize=12)
        plt.ylabel('Reward Media', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_path, f'rewards_epsilon_comparison_{model}_{self.episodes}.png'))
        plt.show()
    
    def plot_scores_epsilon_comparison(self, model, window_size=50):
        """Grafico degli score in funzione di epsilon per un modello.
        
        Per il modello specificato, carica gli score per ciascun epsilon mode e li confronta.
        """
        # Carica gli score per tutte le epsilon mode per il modello specificato
        scores_data = self._load_metrics('score', model)
        
        plt.figure(figsize=(14, 7))
        for i, epsilon_mode in enumerate(self.epsilon_modes):
            if epsilon_mode not in scores_data:
                continue
            
            scores = scores_data[epsilon_mode]
            if len(scores) < window_size:
                continue
            
            # Applica smoothing con media mobile
            smooth_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            
            plt.plot(range(window_size, len(scores)+1),
                     smooth_scores,
                     color=self.colors[i % len(self.colors)],
                     linewidth=2,
                     alpha=0.8,
                     label=f'{model} - {epsilon_mode}')
            
            # Plot della media score per episodio
            mean_score = np.mean(scores)
            plt.plot([0, len(scores)], [mean_score, mean_score],
                     color=self.colors[i % len(self.colors)],
                     linestyle='--',
                     label=f'{model} {epsilon_mode} media')
        
        plt.title(f'Confronto Score per Epsilon Mode ({model})', fontsize=14)
        plt.xlabel('Episodi', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_path, f'scores_epsilon_comparison_{model}_{self.episodes}.png'))
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

    # Generazione grafici per il confronto tra modelli
    plotter.plot_rewards_comparison(window_size=100)
    plotter.plot_scores_comparison(window_size=100)
    
    # Confronto delle epsilon mode per un singolo modello (ad esempio, 'Q-Learning')
    plotter.plot_rewards_epsilon_comparison('Q-Learning', window_size=100)
    plotter.plot_scores_epsilon_comparison('Q-Learning', window_size=50)


if __name__ == "__main__":
    main()