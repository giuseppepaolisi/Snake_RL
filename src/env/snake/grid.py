
from .snake import Snake
from .apple import Apple
from .point import Point
from .const.rewards import REWARD_COLLISION, REWARD_COLLISION_SELF, REWARD_FOOD, REWARD_STEP
from .const.states import VOID, SNAKE, APPLE
from .const.colors import BODY_COLOR, FOOD_COLOR, HEAD_COLOR, SPACE_COLOR
import numpy as np
import random

class Grid:
    """ 
        Definisce la griglia per il gioco Snake.
    """
    def __init__(self, grid_size=(10, 10)) -> None:
        """ Inizializza una griglia

        Args:
            grid_size (tuple, optional): Definisce le dimensioni della griglia di gico. Defaults to (10, 10).
        """
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=np.int32)
        self.snake = Snake(self._place_snake())
        self.apple = Apple(self._place_apple())
        self.score = 0
        self.done = False # Stato del gioco
        self.grid_color = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
        
    def reset(self) -> None:
        """
            Reset della griglia di gioco.
        """
        self.grid = np.zeros(self.grid_size, dtype=np.int32)
        self.snake = Snake(self._place_snake())
        self.apple = Apple(self._place_apple()) 
        self.score = 0
        self.done = False
        self.grid_color = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
        
    def _place_snake(self) -> Point:
        """ Assegna a Snake una posizione casuale all'interno del gioco.

        Returns:
            Point: Ritorna le coordinate del serpente nella griglia
        """
        return Point(random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))

    def _place_apple(self) -> Point:
        """ Posiziona il cibo in una posizione casuale della griglia, evitando il corpo del serpente.

        Returns:
            Point: Ritorna le coordinate della mela nella griglia
        """
        while True:
            apple = Point(random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
            if apple not in self.snake.get_all_body():  # Assicura che il cibo non sia su tutto il corpo del serpente
                return apple
            
    def step(self, direction: Point):
        """ Sposta Snake di un passo nella griglia.

        Args:
            direction (Point): Definisce una direzione sottoforma di punto.

        Returns:
            _type_: Ritorna cone osservazione la griglia di gioco, la ricompensa, dice se il gioco è terminato, il punteggio ottenuto
        """
        self.snake.move(direction)
        
        body = self.snake.get_body()
        head = self.snake.get_head()
        
        # Controlla le collisioni (bordi della griglia)
        if (head.get_x() < 0 or head.get_x() >= self.grid_size[0] or
                head.get_y() < 0 or head.get_y() >= self.grid_size[1]):
            reward = REWARD_COLLISION  # Penalità per collisione
            self.done = True
            return self._get_obs(), reward, self.done, {"score": self.score}
        
        # Collisione corpo
        if(head in body):

            reward = REWARD_COLLISION_SELF  # Penalità per collisione con se stesso
            self.done = True
            return self._get_obs(), reward, self.done, {"score": self.score}

        # Controlla se il cibo è stato mangiat
        if (head == self.apple.get_position()):
            reward = REWARD_FOOD
            self.score += 1
            self.apple = Apple(self._place_apple())  # Genera un nuovo cibo
            self.snake.set_grow(True)
        else:
            self.snake.set_grow(False)    # Il serpente non cresce
            reward = REWARD_STEP         # Penalità per una mossa senza mangiare     

        return self._get_obs(), reward, self.done, {"score": self.score}
        
    def _get_obs(self):
        """
            Genera l'osservazione attuale sotto forma di matrice 2D.
            La griglia è aggiornata con:
                - 1 per le posizioni del corpo del serpente.
                - 2 per la posizione della mela.
        """
        grid = np.zeros(self.grid_size, dtype=np.int32)
        
        snake = self.snake.get_all_body()
        
        # Itera sul corpo del serpente e aggiorna la griglia
        for point in snake:
            x, y = point.get_x(), point.get_y()
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                grid[x, y] = SNAKE

        # Aggiorna la posizione della mela
        apple_x, apple_y = self.apple.get_position().get_x(), self.apple.get_position().get_y()
        if 0 <= apple_x < self.grid_size[0] and 0 <= apple_y < self.grid_size[1]: 
            grid[apple_x, apple_y] = APPLE

        return grid
    
    def get_grid_color(self):
        """ Definisce una matrice con occorrenze array di colori.

        Returns:
            _type_: Ritorna una matrice con occorrenze di array di colori.
        """
        self.grid_color = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
        for point in self.snake.get_body():
            self._update_color_cell(point, BODY_COLOR)
        self._update_color_cell(self.snake.get_head(), HEAD_COLOR)
        self._update_color_cell(self.apple.get_position(), FOOD_COLOR)
        return self.grid_color
    
    def _update_color_cell(self, position: Point, color):
        """Aggiorna il colore di una cella specifica."""
        x, y = position.get_point()
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            self.grid_color[x, y] = color