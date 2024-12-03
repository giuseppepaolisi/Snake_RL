import numpy as np

class QLearningAgent:
    def __init__(self, action_space, observation_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.995):
        """
        Initialize Q-Learning Agent
        
        Args:
            action_space (gym.spaces.Discrete): Action space of the environment
            observation_space (gym.spaces.Box): Observation space of the environment
            learning_rate (float): Learning rate (alpha)
            discount_factor (float): Discount factor (gamma)
            exploration_rate (float): Initial exploration rate
            min_exploration_rate (float): Minimum exploration rate
            exploration_decay_rate (float): Rate of exploration decay
        """
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Hyperparameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = min_exploration_rate
        self.epsilon_decay = exploration_decay_rate
        
        # Q-Table initialization
        self.q_table = {}
    
    def _discretize_state(self, state):
        """
        Convert continuous state to a hashable representation
        
        Args:
            state (np.ndarray): State from the environment
        
        Returns:
            tuple: Discretized state representation
        """
        # Convert state to a tuple for hashing
        return tuple(map(tuple, state))
    
    def get_q_value(self, state, action):
        """
        Get Q-value for a state-action pair
        
        Args:
            state (np.ndarray): Current state
            action (int): Action index
        
        Returns:
            float: Q-value for the state-action pair
        """
        state_key = self._discretize_state(state)
        
        # Initialize Q-value to 0 if state-action pair not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        
        return self.q_table[state_key][action]
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy strategy
        
        Args:
            state (np.ndarray): Current state
        
        Returns:
            int: Selected action
        """
        # Exploration
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        # Exploitation
        state_key = self._discretize_state(state)
        
        # Initialize Q-values for this state if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        
        # Choose action with highest Q-value
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using Q-learning update rule
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode is finished
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Initialize Q-values if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        
        if done:
            # Terminal state
            max_next_q = 0
        else:
            # Max Q-value for next state
            max_next_q = np.max(self.q_table[next_state_key])
        
        # Q-value update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self, filename='q_table.npy'):
        """
        Save Q-table to a file
        
        Args:
            filename (str): Filename to save Q-table
        """
        np.save(filename, self.q_table)
    
    def load_q_table(self, filename='q_table.npy'):
        """
        Load Q-table from a file
        
        Args:
            filename (str): Filename to load Q-table from
        """
        self.q_table = np.load(filename, allow_pickle=True).item()
