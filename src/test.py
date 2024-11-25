import gym

from env.snake.grid import Grid
from env.snake_env import Snake_Env
done = False
i = 0
#while not done:
"""action = env.action_space.sample()  # Random action
obs, reward, done, info = env.step(action)
env.render()
print(f"{i} Reward: {reward}, Score: {env.score}")
i+=1"""
#game = GameEnvironment(grid_size=(10, 10), cell_size=30)
#game.run()

#game = Game(grid_size=(10, 10), cell_size=30)
#game.render()
"""snake = Snake()
action = snake.action_space.sample()
snake.step(action)
snake.render()"""
#env.close()

env = Snake_Env()
i = 0
while i<10000:
    action = env.action_space.sample()
    grid, reward, done, dict =env.step(action)
    env.render()
    print(f"{i} Reward: {reward}, Score: {dict["score"]}")
    i+=1
    if(done):
        env.reset()
