from src.env.agents import SarsaAgent


def discretize_state(state):
    """
    Converte lo stato continuo in uno stato discreto.
    Usa un semplice hash per stati complessi.
    """
    return hash(tuple(state.flatten())) % 1000

def train_agent(env, agent, episodes=500):
    for e in range(episodes):
        state = env.reset()
        state = discretize_state(state)  # Discretizza lo stato iniziale
        action = agent.choose_action(state)  # Prima azione
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state)
            if isinstance(agent, SarsaAgent):
                next_action = agent.choose_action(next_state)  # Azione successiva (SARSA)
                agent.update_q_value(state, action, reward, next_state, next_action, done)
                action = next_action  # Aggiorna l'azione
            else:
                agent.update_q_value(state, action, reward, next_state, done)
                action = next_state  # Avanza senza necessità di una nuova azione per Q-Learning
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        print(f"Episode: {e+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
