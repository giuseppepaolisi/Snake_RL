import numpy as np
import pickle
import os

# Classe base per gli agenti
class BaseAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
    
    def choose_action(self, state):
        raise NotImplementedError("Questo metodo deve essere implementato nella sottoclasse.")

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# Agente Q-Learning
class QLearningAgent(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=0.01, epsilon=1.0, gamma=0.95, **kwargs):
        super().__init__(state_size, action_size, learning_rate=0.01, epsilon=1.0, **kwargs)
        self.gamma = gamma
        
        self.q_table = {}
        
    def get_state_key(self, state):
        """
        Converte la griglia in una key per una hashtable
        """
        return tuple(state.flatten())
    
    def get_q_value(self, state, action):
        """
        Ritorna un Q-value per una coppia stato-azione 
        Se lo stato non è mai stato incontrato lo inizializza
        """
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return self.q_table[state_key][action]
    
    def choose_action(self, state):
        """ Sleziona un'azione usandoi la strategia epsilon-greedy

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Esplora
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key]) # sfrutta

    def update(self, state, action, reward, next_state, done):
        """ Q-Learning update rule
        
        Q(s,a) = Q(s,a) + lr [r + gamma * max(Q(s',a')) - Q(s,a)]

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            done (function): _description_
        """
        current_q = self.get_q_value(state, action)
        
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table.get(
                self.get_state_key(next_state), 
                np.zeros(self.action_size)
            ))
        
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        self.q_table[state_key][action] = new_q
        
        # Decadimento epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def save(self, filepath):
        """Salva il modello"""
        try:
            # Salva sia la Q-table che l'epsilon
            save_data = {
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                'learning_rate' : self.learning_rate,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"Modello salvato con successo in {filepath}")
        except Exception as e:
            print(f"Errore durante il salvataggio: {e}")
    
    def load(self, filepath):
        """carica il modello"""
        if not os.path.exists(filepath):
            print(f"File non trovato: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Carica Q-table
            self.q_table = loaded_data['q_table']
            
            # Ripristina epsilon, se salvato
            if 'epsilon' in loaded_data:
                self.epsilon = loaded_data['epsilon']
            
            if 'gamma' in loaded_data:
                self.gamma = loaded_data['gamma']
                
            if 'learning_rate' in loaded_data:
                self.learning_rate = loaded_data['learning_rate']
            
            print(f"Modello caricato con successo da {filepath}")
            return True
        except Exception as e:
            print(f"Errore durante il caricamento: {e}")
            return False
        