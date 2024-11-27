
import gym
from gym.envs.registration import register

# Registrazione dell'ambiente Snake
register(
    id='Snake-v0',
    entry_point='__main__:Snake_Env',
)

# Creazione dell'ambiente
env = gym.make('Snake-v0')
done = False
i = 0

while i<100:
    ricompensa_cum = 0
    env.reset()
    while not done:
        action = env.action_space.sample()
        grid, reward, done, dict =env.step(action)
        env.render()
        print(f"{i} Reward: {reward}, Score: {dict['score']}")
        ricompensa_cum += reward
    done = False
    print(f"\n\t***episodio: {i} ricompensa {ricompensa_cum}\n")
    i+=1
env.close()