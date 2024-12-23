from gym.envs.registration import register
import gym
from agents import QLearningAgent, Sarsa, DQNAgent
import numpy as np
from train import Train
from metrics_plotter import plot_combined_metric
from env.snake_env import Snake_Env


def main():
    register(
        id='Snake-v0',
        entry_point='env.snake_env:Snake_Env',
        max_episode_steps=300,
    )
    size = 5
    env = Snake_Env(size=size)

    # Dimensione dello stato
    state_size = (
        2       # posizione testa del serpente (x, y)
        + 2     # posizione della mela (x, y)
        + 4     # orientamento one-hot encoding (4 direzioni)
        + 1     # distanza dalla mela
        + 2     # direzione relativa della mela
        + 4     # prossimità ai muri (4 direzioni)
        + 4     # prossimità al proprio corpo (4 direzioni)
    )
    
    action_size = env.action_space.n
    episodes=10000
    max_steps=200
    
    # Create and train DQN agent
    #dqn_agent = DQNAgent(state_size, action_size, learning_rate=0.01, gamma=0.95, epsilon=0.9, episodes=episodes)
    #dqn_train = Train(env, dqn_agent, episodes, max_steps=max_steps)
    #dqn_train.train()

    '''sarsa_agent = Sarsa(state_size, action_size, learning_rate=0.01, gamma=0.95, epsilon=0.9, episodes=episodes)
    #agent.load('models/snake_q_agent.pkl')
    sarsa_train = Train(env, sarsa_agent, episodes, max_steps=max_steps)
    sarsa_train.train()'''
    
    q_learning_agent = QLearningAgent(state_size, action_size, learning_rate=0.01, gamma=0.99, epsilon=0.99, episodes=episodes)
    q_learning_agent_train = Train(env, q_learning_agent, episodes, max_steps=max_steps)
    q_learning_agent_train.train()

    #q_learning_agent.load(f'models/snake_{q_learning_agent.get_model()}_{episodes}.pkl')
    
    # Dopo l'addestramento di entrambi gli agenti, genera grafico combinato per lo score
    '''plot_combined_metric(
        metric='score',
        episode=episodes,
        agents=[(sarsa_agent, 'Sarsa'), (q_learning_agent, 'Q-Learning')]
    )'''
    env.close()
    env = Snake_Env(size=size, render_mode='human')
    num_episodes = 10
    agent = q_learning_agent
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        print(f'--- Episode {episode + 1} ---')

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)  # Ignoriamo truncated
            total_reward += reward
            state = next_state
            
            if done:
                break

        print(f'Total reward for episode {episode + 1}: {total_reward} Score: {info["score"]}')
    
    env.close()
    
if __name__ == "__main__":
    main()