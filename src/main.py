from gym.envs.registration import register
import gym
from agents import QLearningAgent, Sarsa
import numpy as np
from train import Train
def main():
    register(
        id='Snake-v0',
        entry_point='env.snake_env:Snake_Env',
        max_episode_steps=300,
    )
    env = gym.make('Snake-v0', size=5)

    state_size = env.observation_space
    action_size = env.action_space.n
    episodes=1000
    max_steps=200

    sarsa_agent = Sarsa(state_size, action_size, learning_rate=0.01, gamma=0.95, epsilon=0.9, episodes=episodes)
    #agent.load('models/snake_q_agent.pkl')
    sarsa_train = Train(env, sarsa_agent, episodes, max_steps=max_steps)
    sarsa_train.train()
    
    
    q_learning_agent = QLearningAgent(state_size, action_size, learning_rate=0.01, gamma=0.95, epsilon=0.9, episodes=episodes)
    q_learning_agent_train = Train(env, q_learning_agent, episodes, max_steps=max_steps)
    q_learning_agent_train.train()

    #agent.load(f'models/snake_q_agent_{episodes}.pkl')
    
    #Esecuzione di entrambi gli agent per testare le loro performance
    for agent, agent_name in zip ([sarsa_agent, q_learning_agent], ['Sarsa', 'Q-Learning']):
        env = gym.make('Snake-v0', render_mode='human')
        num_episodes = 10
        for episode in range(num_episodes):
            state, info = env.reset()
            total_reward = 0
            done = False

            print(f"--- Episodio {episode + 1} ---")

            while True:
            #for step in range(max_steps):
                action = agent.choose_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break

            print(f"Totale ricompensa per l'episodio {episode + 1} con {agent_name}: {total_reward} Score: {info['score']}")

    env.close()
    
if __name__ == "__main__":
    main()