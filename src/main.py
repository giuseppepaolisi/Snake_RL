import gym
from env.agents import QLearningAgent
import numpy as np
from train import Train
from gym.envs.registration import register

def main():
    # Registriamo l'ambiente direttamente nel main.py
    register(
        id='Snake-v0',
        entry_point='env.snake_env:Snake_Env',
    )

    # Creazione dell'ambiente Snake
    env = gym.make('Snake-v0')

    grid_size = [5, 5]
    state_size = env.observation_space
    action_size = env.action_space.n
    episodes=200000
    
    agent = QLearningAgent(state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=0.1)
    
    train = Train(env,agent, episodes, max_steps=200)
    train.train()
    
    #agent.load('models/snake_q_agent_200.pkl')
    
    num_episodes = 10

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        print(f"--- Episodio {episode + 1} ---")

        while not done:
            env.render() 
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

        print(f"Totale ricompensa per l'episodio {episode + 1}: {total_reward} Score: {info['score']}")

    env.close()
    

if __name__ == "__main__":
    main()
