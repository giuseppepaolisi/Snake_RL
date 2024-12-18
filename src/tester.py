from env.snake_env import Snake_Env
from agents import QLearningAgent, Sarsa, DQN

def test_agent(agent, num_episodes=10, size=5):
    env = Snake_Env(size=size, render_mode='human')
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        print(f"--- Episode {episode + 1} ---")

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            
            if done:
                break

        print(f"Total reward for episode {episode + 1}: {total_reward} Score: {info['score']}")
    
    env.close()