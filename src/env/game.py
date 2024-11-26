import pygame
import random
import numpy as np
from .snake.colors import BODY_COLOR, FOOD_COLOR, HEAD_COLOR, SPACE_COLOR

class Game():
    """
        Gestisce la grafica del gioco Snake.
    """
    
    def __init__(self, grid_size=(10, 10), cell_size=20):
        """ Inizializza la grafica di gioco.

        Args:
            grid_size (tuple, optional): Dimensione della griglia di gioco. Defaults to (10, 10).
            cell_size (int, optional): Dimensione delle celle del gioco. Defaults to 20.
        """
        pygame.init()
        
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid = np.zeros(self.grid_size, dtype=np.int32)
        
        self.screen = pygame.display.set_mode((grid_size[1] * cell_size, grid_size[0] * cell_size))
        self.clock = pygame.time.Clock()
        self.running = True
        self.score = 0
    
    def render(self, grid_color):
        """ Aggiorna la finestra di gioco.

        Args:
            grid_color (_type_): Grigli con i colori come occorrenze.
        """
        self.grid = grid_color

        # Renderizza la griglia sulla finestra
        self.screen.fill(SPACE_COLOR)  # Sfondo
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                color = grid_color[x, y]
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)

        pygame.display.flip()
        self.clock.tick(5)