import pickle
import os
import random
from collections import deque
import numpy as np
import torch
from torch import nn, optim
import math

# Classe base per gli agenti
class BaseAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, episodes=1000, size=5, gamma=0.95, decay_mode="exponential"):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.size = size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.epsilon_start = epsilon
        self.q_table = {}  # Inizializza la q_table
        self.decay_mode = decay_mode

    def choose_action(self, state):
        raise NotImplementedError('Questo metodo deve essere implementato nella sottoclasse.')
    
    def decay_epsilon(self, episodes_completed=0, step_size=10):
        """Decremento epsilon secondo diverse strategie"""
        if self.decay_mode == 'exponential':
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        elif self.decay_mode == 'linear':
            decay_rate = (self.epsilon_start - self.epsilon_min) / self.episodes
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_rate)
        elif self.decay_mode == 'cosine':
            progress = episodes_completed / self.episodes
            self.epsilon = self.epsilon_min + 0.5 * (self.epsilon_start - self.epsilon_min) * (1 + math.cos(math.pi * progress))
        elif self.decay_mode == 'step':
            if episodes_completed % step_size == 0:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
    def save(self, filename):
        'Salva il modello'
        try:
            # Salva sia la Q-table che l'epsilon
            save_data = {
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                'learning_rate': self.learning_rate,
            }
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
            print(f'Modello salvato con successo in {filename}')
        except (IOError, RuntimeError) as e:
            print(f'Errore durante il salvataggio: {e}')

    def load(self, filename):
        'Carica il modello'
        if not os.path.exists(filename):
            print(f'File non trovato: {filename}')
            return False
        try:
            with open(filename, 'rb') as f:
                loaded_data = pickle.load(f)
            # Carica Q-table
            self.q_table = loaded_data.get('q_table', {})
            # Ripristina epsilon, se salvato
            self.epsilon = loaded_data.get('epsilon', self.epsilon)
            # Ripristina gamma, se salvato
            self.gamma = loaded_data.get('gamma', self.gamma)
            # Ripristina learning_rate, se salvato
            self.learning_rate = loaded_data.get('learning_rate', self.learning_rate)
            print(f'Modello caricato con successo da {filename}')
            return True
        except (IOError, RuntimeError) as e:
            print(f'Errore durante il caricamento: {e}')
            return False

    def get_model(self) -> str:
        return ''

# Agente Q-Learning
class QLearningAgent(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=0.01, epsilon=0.99, epsilon_decay=0.995, epsilon_min=0.01, episodes=1000, size=5, gamma=0.95, decay_mode="exponential"):
        super().__init__(state_size, action_size, learning_rate, epsilon, epsilon_decay, epsilon_min, episodes, size, gamma, decay_mode)

    def get_model(self):
        return 'Q-Learning'

    def get_state_key(self, state):
        """
        Converte lo stato in una chiave univoca per la Q-table.
        Lo stato include coordinate del serpente, coordinate della mela, orientamento, distanza, se è vicino al bordo e se è vicino al corpo
        """
        snake_head = tuple(state['snake'][0])
        apple = tuple(state['apple'])
        orientation = state['orientation'].index(1)
        distance = int(state['distance_to_apple'][0] * 10)  # Discretizza la distanza
        relative_direction = tuple(np.round(state['relative_direction'], 1))
        proximity_to_wall = tuple(state['proximity_to_wall'])
        body_proximity = tuple(state['body_proximity'])
        return (snake_head, apple, orientation, distance, relative_direction, proximity_to_wall, body_proximity)

    def get_q_value(self, state, action):
        """
        Ritorna un Q-value per una coppia stato-azione
        Se lo stato non è mai stato incontrato lo inizializza
        """
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-1, 1, self.action_size)
        return self.q_table[state_key][action]

    def choose_action(self, state):
        """ Seleziona un'azione usando la strategia epsilon-greedy """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Esplora
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-1, 1, self.action_size)
        return np.argmax(self.q_table[state_key])  # Sfrutta

    def update(self, state, action, reward, next_state, done):
        """ Q-Learning update rule """
        current_q = self.get_q_value(state, action)
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table.get(
                self.get_state_key(next_state),
                np.random.uniform(-1, 1, self.action_size)
            ))
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-1, 1, self.action_size)
        self.q_table[state_key][action] = new_q
        

class Sarsa(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=0.01, epsilon=0.99, epsilon_decay=0.995, epsilon_min=0.01, episodes=1000, size=5, gamma=0.95, decay_mode="exponential"):
        super().__init__(state_size, action_size, learning_rate, epsilon, epsilon_decay, epsilon_min, episodes, size, gamma, decay_mode)

    def get_model(self):
        return 'SARSA'

    def get_state_key(self, state):
        """
        Converte lo stato in una chiave univoca per la Q-table.
        Lo stato include coordinate del serpente, coordinate della mela, orientamento, distanza, se è vicino al bordo e se è vicino al corpo
        """
        snake_head = tuple(state['snake'][0])
        apple = tuple(state['apple'])
        orientation = state['orientation'].index(1)
        distance = int(state['distance_to_apple'][0] * 10)  # Discretizza la distanza
        relative_direction = tuple(np.round(state['relative_direction'], 1))
        proximity_to_wall = tuple(state['proximity_to_wall'])
        body_proximity = tuple(state['body_proximity'])
        return (snake_head, apple, orientation, distance, relative_direction, proximity_to_wall, body_proximity)

    def get_q_value(self, state, action):
        """
        Ritorna un Q-value per una coppia stato-azione
        Se lo stato non è mai stato incontrato lo inizializza
        """
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-1, 1, self.action_size)
        return self.q_table[state_key][action]

    def choose_action(self, state):
        """ Seleziona un'azione usando la strategia epsilon-greedy """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Esplora
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-1, 1, self.action_size)
        return np.argmax(self.q_table[state_key])  # Sfrutta

    def update(self, state, action, reward, next_state, done):
        """ Sarsa update rule """
        current_q = self.get_q_value(state, action)
        if done:
            next_q = 0
        else:
            next_action = self.choose_action(next_state)
            next_q = self.get_q_value(next_state, next_action)
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_q - current_q)
        state_key = self.get_state_key(state)
        self.q_table[state_key][action] = new_q
        

class DQN(nn.Module):
    """Architettura Deep Q-Network"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """Passaggio in avanti attraverso la rete."""
        return self.network(x)

class DQNAgent(BaseAgent):
    """DQN Agent implementation."""
    def __init__(self, state_size, action_size, hidden_size=64, memory_size=10000, batch_size=32, target_update=10, learning_rate=0.01, epsilon=0.99, epsilon_decay=0.995, epsilon_min=0.01, episodes=1000, size=5, gamma=0.95, decay_mode="exponential"):
        super().__init__(state_size, action_size, learning_rate, epsilon, epsilon_decay, epsilon_min, episodes, size, gamma, decay_mode)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0
        # Inizializzazione della rete
        self.policy_net = DQN(state_size, hidden_size, action_size)
        self.target_net = DQN(state_size, hidden_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def get_model(self):
        return 'DQN'

    def get_state_tensor(self, state):
        """ Converte lo stato dell'ambiente in un tensore per la rete DQN."""
        state_features = []
        # Posizione testa del serpente (x, y) normalizzata
        snake_head = state['snake'][0]
        state_features.extend([snake_head[0] / (self.size - 1), snake_head[1] / (self.size - 1)])
        # Posizione della mela (x, y) normalizzata
        apple = state['apple']
        state_features.extend([apple[0] / (self.size - 1), apple[1] / (self.size - 1)])
        # Codifica one-hot dell'orientamento
        orientation = state['orientation']
        state_features.extend(orientation)
        # Distanza dalla mela normalizzata
        max_distance = np.sqrt(2 * (self.size - 1)**2)
        distance = state['distance_to_apple'][0] / max_distance
        state_features.append(distance)
        # Direzione relativa (dx, dy) già normalizzata
        relative_direction = state['relative_direction']
        state_features.extend(relative_direction)
        # Prossimità ai muri (4 flag binarie)
        proximity_to_wall = state['proximity_to_wall']
        state_features.extend(proximity_to_wall)
        # Prossimità del corpo (4 flag binarie)
        body_proximity = state['body_proximity']
        state_features.extend(body_proximity)
        # Verifica che le feature dello stato siano della dimensione attesa
        expected_size = 19  # 2+2+4+1+2+4+4
        if len(state_features) != expected_size:
            raise ValueError(f"Dimensione delle feature dello stato incompatibile: atteso {expected_size}, ottenuto {len(state_features)}")
        return torch.FloatTensor(state_features).to(self.device)

    def choose_action(self, state):
        """ Seleziona un'azione usando la politica epsilon-greedy """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = self.get_state_tensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        """ Aggiorna la rete DQN con l'esperienza """
        # Memorizza la transizione nella memoria
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return
        # Campiona un batch casuale dalla memoria
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Converte in tensori
        state_batch = torch.stack([self.get_state_tensor(s) for s in states]).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.stack([self.get_state_tensor(s) for s in next_states]).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        # Calcola i valori Q correnti
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        # Calcola i valori Q successivi
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        # Calcola la perdita e aggiorna
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        # Discesa del gradiente
        self.optimizer.zero_grad()
        loss.backward()
        # Taglia i gradienti per prevenire l'esplosione dei gradienti
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        # Aggiorna la rete target periodicamente
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps += 1

    def save(self, filename):
        """Salva lo stato dell'agente."""
        try:
            save_data = {
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps,
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(save_data, filename)
            print(f'Successfully saved model to {filename}')
        except (IOError, RuntimeError) as e:
            print(f'Failed to save model: {str(e)}')

    def load(self, filename):
        """Carica il modello"""
        if not os.path.exists(filename):
            print(f"File non trovato: {filename}")
            return False
        try:
            save_data = torch.load(filename)
            self.policy_net.load_state_dict(save_data['policy_net'])
            self.target_net.load_state_dict(save_data['target_net'])
            self.epsilon = save_data.get('epsilon', self.epsilon)
            self.steps = save_data.get('steps', self.steps)
            self.optimizer.load_state_dict(save_data['optimizer'])
            print(f"Modello caricato con successo da {filename}")
            return True
        except (IOError, RuntimeError) as e:
            print(f"Errore durante il caricamento: {e}")
            return False