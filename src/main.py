from gym.envs.registration import register
import gym
from env.agents import QLearningAgent
import numpy as np
from train import Train
def main():
    register(
        id='Snake-v0',
        entry_point='env.snake_env:Snake_Env',
        max_episode_steps=300,
    )
    env = gym.make('Snake-v0')

    state_size = env.observation_space
    action_size = env.action_space.n
    episodes=50000

    agent = QLearningAgent(state_size, action_size, learning_rate=0.01, gamma=0.95, epsilon=0.3)
    max_steps=200
    #agent.load('models/snake_q_agent.pkl')

    train = Train(env, agent, episodes, max_steps=max_steps)
    train.train()

    #agent.load(f'models/snake_q_agent_{episodes}.pkl')
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

        print(f"Totale ricompensa per l'episodio {episode + 1}: {total_reward} Score: {info['score']}")

    env.close()
    
if __name__ == "__main__":
    main()