import numpy as np

# Classe base per gli agenti
class BaseAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def choose_action(self, state):
        raise NotImplementedError("Questo metodo deve essere implementato nella sottoclasse.")

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# Agente Q-Learning
class QLearningAgent(BaseAgent):
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Esplora
        return np.argmax(self.q_table[state])  # Sfrutta

    def update_q_value(self, state, action, reward, next_state, done):
        max_next_q = np.max(self.q_table[next_state]) if not done else 0
        self.q_table[state, action] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[state, action])

# Agente SARSA
class SarsaAgent(BaseAgent):
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Esplora
        return np.argmax(self.q_table[state])  # Sfrutta

    def update_q_value(self, state, action, reward, next_state, next_action, done):
        next_q = self.q_table[next_state, next_action] if not done else 0
        self.q_table[state, action] += self.alpha * (reward + self.gamma * next_q - self.q_table[state, action])
