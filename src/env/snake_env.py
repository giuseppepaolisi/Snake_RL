from typing import List, Tuple
import numpy as np
import gym
from gym import error, spaces, utils
from .snake.grid import Grid
from .snake.actions import UP, RIGHT, DOWN, LEFT, DIRECTION_UP, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_LEFT
from .game import Game

class Snake_Env(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_size=[10,10], cell_size=30):
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        # Spazio delle azioni (4 direzioni: 0=Su, 1=Destra, 2=Giù, 3=Sinistra)        
        self.action_space = spaces.Discrete(4)
        # Spazio delle osservazioni: griglia con valori 0 (vuoto), 1 (testa del serpente), 2 (cibo)
        self.observation_space = spaces.Box(low=0, high=2, shape=self.grid_size, dtype=np.int32)
        
        self.grid = Grid(grid_size)
        
        self.game = Game(grid_size, cell_size)
        
    def step(self, action: np.int32):
        """ Permette di effettuare un azione nel gioco.

        Args:
            action (np.int32): Azione scelta (0=Su, 1=Destra, 2=Giu, 3=Sinistra).

        Returns:
            _type_: Ritorna cone osservazione la griglia di gioco, la ricompensa, dice se il gioco è terminato, il punteggio ottenuto
        """
        direction = None
        if action == UP:  # Su
            direction = DIRECTION_UP
        elif action == RIGHT:  # Destra
            direction = DIRECTION_RIGHT
        elif action == DOWN:  # Giù
            direction = DIRECTION_DOWN
        elif action == LEFT:  # Sinistra
            direction = DIRECTION_LEFT
        self.last_obs, reward, done, info = self.grid.step(direction)
        return  self.last_obs, reward, done, info
    
    def reset(self):
        self.grid.reset()
        return self.grid._get_obs()
    
    def render(self):
        self.game.render(self.grid.get_grid_color())