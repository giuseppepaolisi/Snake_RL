
from env.snake_env import Snake_Env

done = False
env = Snake_Env()
i = 0

while i<100:
    ricompensa_cum = 0
    while not done:
        action = env.action_space.sample()
        grid, reward, done, dict =env.step(action)
        env.render()
        print(f"{i} Reward: {reward}, Score: {dict['score']}")
        ricompensa_cum += reward
    done = False
    env.reset()
    print(f"\n\t***episodio: {i} ricompensa {ricompensa_cum}\n")
    i+=1
env.close()