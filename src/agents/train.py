import gym
import numpy as np
from ..env.snake_env import Snake_Env
from agent import QLearningAgent

def train(env, agent, num_episodes=1000, max_steps_per_episode=1000):
    """
    Train the Q-learning agent in the Snake environment
    
    Args:
        env (gym.Env): Snake environment
        agent (QLearningAgent): Q-learning agent
        num_episodes (int): Number of training episodes
        max_steps_per_episode (int): Maximum steps allowed per episode
    
    Returns:
        list: List of rewards per episode
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            # Choose action
            action = agent.choose_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # End episode if done
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}, Exploration: {agent.epsilon:.2f}")
    
    return episode_rewards

def main():
    # Create environment
    env = Snake_Env(grid_size=[5, 5], cell_size=30)
    
    # Create agent
    agent = QLearningAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        min_exploration_rate=0.01,
        exploration_decay_rate=0.995
    )
    
    # Train agent
    rewards = train(env, agent, num_episodes=2000, max_steps_per_episode=500)
    
    # Save Q-table
    agent.save_q_table('snake_q_table.npy')
    
    # Optional: Visualize final performance
    test_episodes = 10
    test_rewards = []
    for _ in range(test_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        test_rewards.append(total_reward)
    
    print(f"Test Performance - Mean Reward: {np.mean(test_rewards):.2f}, Std: {np.std(test_rewards):.2f}")

if __name__ == "__main__":
    main()