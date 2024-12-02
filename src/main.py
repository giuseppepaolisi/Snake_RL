import gym
from env.agents import QLearningAgent, SarsaAgent
from env.train import train_agent
from env.snake.Snake_Env import Snake_Env
from src.env.train import discretize_state

def main():
    # Registriamo l'ambiente direttamente nel main.py
    gym.envs.registration.register(
        id='Snake-v0',  # Nome dell'ambiente
        entry_point='env.snake.snake_env:Snake_Env',  # Il percorso completo della classe Snake_Env
        max_episode_steps=100,  # Numero massimo di passi per episodio
    )

    # Creazione dell'ambiente Snake
    env = gym.make('Snake-v0')  # Gym ora caricherà Snake_Env registrato

    state_size = 1000  # Numero massimo di stati discreti
    action_size = env.action_space.n

    # Lista di modelli da testare
    agents = [
        ("Q-Learning", QLearningAgent(state_size=state_size, action_size=action_size, alpha=0.1, gamma=0.95)),
        ("SARSA", SarsaAgent(state_size=state_size, action_size=action_size, alpha=0.1, gamma=0.95))
    ]
    
    # Testa tutti i modelli
    for model_name, agent in agents:
        print(f"\n\n*** Inizio Training per il modello: {model_name} ***")
        
        # Training
        train_agent(env, agent, episodes=100)

        # Testing
        print(f"\nTesting {model_name} Agent...")
        state = env.reset()
        state = discretize_state(state)
        done = False
        while not done:
            action = agent.choose_action(state)
            state, _, done, _ = env.step(action)
            env.render()

if __name__ == "__main__":
    main()
