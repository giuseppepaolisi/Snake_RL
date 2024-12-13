import numpy as np
import matplotlib.pyplot as plt
import os
from agents import QLearningAgent, Sarsa

def plot_combined_metric(metric, episode, labels):
    
    plt.figure(figsize=(12, 6))

    for label in labels:
        # Percorso del file delle metriche
        rewards_path = f'metrics/{metric}_{label}_{episode}.npy'
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
    
def main():
    # Definisci il numero di episodi
    episodes = 1000  # Cambia questo valore in base alle tue esigenze

    models = ["Q-Learning", "SARSA"]
    
    # Inizializza una lista per le metriche
    agents_metrics = {metric: [] for metric in ['score', 'total_rewards', 'epsilon_decay']}

    # Carica le metriche per ogni modello
    for model in models:
        for metric in agents_metrics.keys():
            # Usa i nomi delle metriche in minuscolo
            file_path = f'metrics/{metric.lower()}_{model}_{episodes}.npy'
            print(f"Trying to load: {file_path}")  # Stampa il percorso del file
            try:
                # Carica la metrica specifica
                values = np.load(file_path)  # Assicurati che file_path sia una stringa valida
                agents_metrics[metric].append((model, values))  # Aggiungi la metrica alla lista

            except FileNotFoundError:
                print(f"File non trovato per il modello: {model}, metrica: {metric}")
            except Exception as e:
                print(f"Errore durante il caricamento del file: {e}")

    # Stampa le metriche utilizzando il metodo di plotting
    for metric, data in agents_metrics.items():
        if data:  # Se ci sono dati per la metrica
            # Passa i dati corretti alla funzione
            labels = [agent[0] for agent in data]  # Estrai solo i nomi dei modelli
            plot_combined_metric(metric=metric, episode=episodes, labels=labels)

if __name__ == "__main__":
    main()