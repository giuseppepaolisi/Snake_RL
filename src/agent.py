import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        """ 

        Args:
            input_dim (_type_): dimensione dello stato.
            output_dim (_type_): azioni possibili sull'ambiente
        """
        super(DQN, self).__init__()
        # Definizione della rete neurale
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)
    
# gamma, start_epsilon_decay, end_epsilon_decay, epsilon_decay_value
class Agent:
    def __init__(self, state_size, action_size, learning_rate = 0.001, gamma = 0.95, epsilon = 0.3, episodes = 100, model_path=None) -> None:
        """ Costruttore agente.

        Args:
            state_size (_type_): Dimensione dello spazio degli stati.
            action_size (_type_): Numero di azioni possibili.
            learning_rate (float, optional): Tasso di apprendimento usato dall'ottimizzatore. Defaults to 0.3.
            gamma (float, optional): Fattore di sconto. Defaults to 0.95.
            epsilon (float, optional): Parametro per l'epsilon-greedy. Defaults to 0.3.
            episodes (int, optional): _description_. Defaults to 100.
            model_path (_type_, optional): Path di un modello esistente. Defaults to None.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_end = 0.01 # Valore minimo di epsilon
        self.epsilon_decay_value = episodes // 2 # Tasso con cui epsilon diminuisce nel tempo
        
        # Apprendimento
        self.q_network = DQN(state_size, action_size) # Viene aggiornata ad ogni passo del training
        
        # Carica un modello pre-addestrato se specificato
        if model_path is not None:
            self.q_network.load_state_dict(torch.load(model_path))
            print(f"Modello caricato da {model_path}")
            
        self.target_network = DQN(state_size, action_size) # Copia i pesi dalla rete q_network e stabilizza il training
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate) # Ottimizzazione.
        self.memory = deque(maxlen=20000)  # Replay buffer
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss più stabile
    
    def act(self, state) -> None:
        """ Policy: con probabilità epsilon-greedy sceglie un'azione casuale

        Args:
            state (_type_): Stato attuale

        Returns:
            _type_: ritorna o un'azione random o l'azione greedy
        """
        # Decadimento esponenziale dell'epsilon
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon_start * (self.epsilon_end / self.epsilon_start) ** (1 / self.epsilon_decay_value)
        )
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        """ Salva l'esperienza nella replay memory

        Args:
            state (_type_): Stato
            action (_type_): Azione
            reward (_type_): Ricompensa
            next_state (_type_): Stato successivo
            done (function): _description_
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """ Memorizza e gestisce le esperienze dell'agente durante l'interazione con l'ambiente.

        Args:
            batch_size (_type_): Determinare il numero di esperienze che vengono estratte dal buffer di memoria e utilizzate per un singolo passo di addestramento della rete neurale.
        """
        if len(self.memory) < batch_size:
            return

        # Campiona casualmente batch di esperienze.
        # Riduce la correlazione tra le esperienze successive.
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Calcolo Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Calcolo loss con Huber Loss
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        # Propaga i gradienti con loss.backward() e aggiorna i pesi.
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """ 
            Copia i pesi della rete q_network nella rete target_network
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path: str):
        """
        Salva un modello

        Args:
            path (str): File path in cui salvare il modello
        """
        torch.save(self.q_network.state_dict(), path)
        print(f"Modello salvato in {path}")
    
    def load_model(self, path: str):
        """
        Caricamento modello pre-addestrato

        Args:
            path (str): File path in cui è salvato il modello
        """
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"Modello caricato da {path}")