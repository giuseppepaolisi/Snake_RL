import gym
from gym import spaces
import pygame
import numpy as np
from .snake.const.rewards import REWARD_COLLISION, REWARD_COLLISION_SELF, REWARD_FOOD, REWARD_STEP_TO_FOOD, REWARD_STEP_NO_FOOD
from .snake.const.actions import FORWARD, RIGHT, LEFT, ORIENTATION_UP, ORIENTATION_RIGHT, ORIENTATION_DOWN, ORIENTATION_LEFT, DIRECTION_UP, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_LEFT
from .snake.const.colors import BODY_COLOR, FOOD_COLOR, HEAD_COLOR, SPACE_COLOR
from .snake.point import Point

class GameGraphics:
    """Gestisce il rendering del gioco usando Pygame."""
    def __init__(self, size, window_size):
        """ 

        Args:
            size (int): Dimensione della griglia (size * size)
            window_size (int): Dimensione della finestra di gioco (pixel)
        """
        self.size = size
        self.window_size = window_size
        self.pix_square_size = window_size / size
        self.window = None
        self.clock = None

    def initialize(self):
        """Inizzializza il rendering del gioco """
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def render(self, snake_body, apple_location, fps):
        """Disegna il serpente, la mela e la griglia sullo schermo."""
        if self.window is None:
            self.initialize()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(SPACE_COLOR)

        # Disegna la mela
        pygame.draw.rect(
            canvas,
            FOOD_COLOR,
            pygame.Rect(
                self.pix_square_size * np.array(apple_location.get_point()),
                (self.pix_square_size, self.pix_square_size),
            ),
        )

        # Disegna il corpo del serpente
        for segment in snake_body[1:]:
            pygame.draw.rect(
                canvas,
                BODY_COLOR,
                pygame.Rect(
                    self.pix_square_size * np.array(segment.get_point()),
                    (self.pix_square_size, self.pix_square_size),
                ),
            )

        # Disegna la testa del serpente
        pygame.draw.rect(
            canvas,
            HEAD_COLOR,
            pygame.Rect(
                self.pix_square_size * np.array(snake_body[0].get_point()),
                (self.pix_square_size, self.pix_square_size),
            ),
        )

        # Disegna la griglia
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),  # Nero
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),  # Nero
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.window_size),
                width=1,
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(fps)

    def close(self):
        """Chiude le risorse """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class Snake_Env(gym.Env):
    """ Inizializza l'ambiente del gioco Snake.

    Args:
        render_mode (str): Modalità di rendering, può essere "human" o "rgb_array"
        size (int): Dimensione della griglia del gioco (size x size).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512
        self.render_mode = render_mode

        # Definizione dello spazio delle osservazioni. Include:
            # 1. La posizione del serpente come lista di segmenti (coordinate x, y)
            # 2. La posizione della mela come una coppia di coordinate (x, y).
        self.observation_space = spaces.Dict({
            "snake": spaces.Box(low=0, high=size - 1, shape=(size*size, 2), dtype=int),
            "apple": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int),
            "orientation": spaces.Discrete(4),  # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        })
        
        # Spazio delle azioni (forward=0, right=1, left=2)
        self.action_space = spaces.Discrete(3)

        # Stato iniziale del gioco
        self.snake_body = []
        self.direction = DIRECTION_RIGHT
        self.orientation = ORIENTATION_RIGHT
        self.apple_location = None
        self.score = 0
        self.graphics = GameGraphics(size, self.window_size)
        
        # Traccia della distanza iniziale per calcolare reward
        self.initial_distance = None

    def _generate_snake(self):
        """Genera il serpente in una posizione casuale."""
        head = Point(
            *self.np_random.integers(0, self.size, size=2)
        )
        self.snake_body = [head]
        for _ in range(1):
            new_segment = Point(
                *np.clip(
                    np.array(head.get_point()) - np.array(self.direction.get_point()),
                    0,
                    self.size - 1,
                )
            )
            self.snake_body.append(new_segment)

    def _generate_apple(self):
        """Genera una mela in una posizione casuale non occupata dal serpente."""
        while True:
            apple = Point(
                *self.np_random.integers(0, self.size, size=2)
            )
            if apple not in self.snake_body:
                self.apple_location = apple
                break

    def reset(self, seed=None, options=None):
        """
        Reset dell'ambiente. Da usare prima di ogni episodio
        
        Args:
            seed (_type_): _description_
            options (_type_): _description_

        Returns:
            Una tupla contenente le osservazioni iniziali e le informazioni aggiuntive.
        """
        super().reset(seed=seed, options=options)
        self.orientation = ORIENTATION_RIGHT
        self.direction = DIRECTION_RIGHT
        self._generate_snake()
        self._generate_apple()
        self.score = 0
        
        # Calcolo distanza iniziale per calcolare reward
        self.initial_distance = self._calculate_distance()
        
        if self.render_mode == "human":
            self.graphics.render(self.snake_body, self.apple_location, self.metadata["render_fps"])
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        """ Genera l'osservazione attuale del gioco

        Returns:
            dict: Include la posizione del serpente, la posizione della mela e l'orientamento del serpente.
        """
        snake_positions = np.array([segment.get_point() for segment in self.snake_body], dtype=np.int32)
        current_length = len(snake_positions)
        
        # Create padded array filled with zeros
        padded_snake = np.zeros((self.size * self.size, 2), dtype=np.int32)
        
        # Fill in actual snake positions
        padded_snake[:current_length] = snake_positions
        
        # Mappa dell'orientamento per convertire in un intero
        orientation_to_int = {
            ORIENTATION_UP: 0,
            ORIENTATION_RIGHT: 1,
            ORIENTATION_DOWN: 2,
            ORIENTATION_LEFT: 3
        }
        
        
        return {
            "snake": padded_snake,
            "apple": np.array(self.apple_location.get_point()),
            "orientation": orientation_to_int[self.orientation],
        }

    def _get_info(self):
        """ Genera informazioni aggiuntive sullo stato del gioco

        Returns:
            dict: Include il punteggio attuale.
        """
        return {"score": self.score}

    def step(self, action):
        """ Permette di effettuare un azione nel gioco.

        Args:
            action (np.int32): Azione scelta (0=Avanti, 1=Destra, 2=Giu).

        Returns:
            _type_: Ritorna cone osservazione la griglia di gioco, la ricompensa, dice se il gioco è terminato, il punteggio ottenuto
        """
        # Mappatura delle azioni in base all'orientamento corrente
        orientations_map = {
            ORIENTATION_UP: {
                FORWARD: DIRECTION_UP,
                RIGHT: DIRECTION_RIGHT,
                LEFT: DIRECTION_LEFT
            },
            ORIENTATION_RIGHT: {
                FORWARD: DIRECTION_RIGHT,
                RIGHT: DIRECTION_DOWN,
                LEFT: DIRECTION_UP
            },
            ORIENTATION_DOWN: {
                FORWARD: DIRECTION_DOWN,
                RIGHT: DIRECTION_LEFT,
                LEFT: DIRECTION_RIGHT
            },
            ORIENTATION_LEFT: {
                FORWARD: DIRECTION_LEFT,
                RIGHT: DIRECTION_UP,
                LEFT: DIRECTION_DOWN
            }
        }

        # Calcola la nuova direzione in base all'azione e all'orientamento corrente
        direction = orientations_map[self.orientation][action]

        # Calcola la nuova posizione della testa.
        new_head = self.snake_body[0] + direction

        # Collisioni corpo
        if new_head in self.snake_body:
            return self._get_obs(), REWARD_COLLISION_SELF, True, False, self._get_info()

        # Collisione griglia
        if not (0 <= new_head.get_x() < self.size and 0 <= new_head.get_y() < self.size):
            return self._get_obs(), REWARD_COLLISION, True, False, self._get_info()

        self.snake_body.insert(0, new_head)
        
         # Aggiorna l'orientamento in base alla nuova direzione
        if direction == DIRECTION_UP:
            self.orientation = ORIENTATION_UP
        elif direction == DIRECTION_RIGHT:
            self.orientation = ORIENTATION_RIGHT
        elif direction == DIRECTION_DOWN:
            self.orientation = ORIENTATION_DOWN
        elif direction == DIRECTION_LEFT:
            self.orientation = ORIENTATION_LEFT
        
        # Calcola distanza dopo il movimento
        final_distance = self._calculate_distance()

        # Controlla se il cibo è stato mangiat
        if new_head == self.apple_location:
            self.score += 1
            reward = REWARD_FOOD
            self._generate_apple()
        else:
            # Controlla se si sta avvicinando al cibo
            if final_distance < self.initial_distance:
                reward = REWARD_STEP_TO_FOOD # Ricompensa per ogni passo effettuato nella direzione del cibo.
            else:
                reward = REWARD_STEP_NO_FOOD
            self.initial_distance = final_distance # Penalità per ogni passo effettuato senza andare nella direzione del cibo.
            self.snake_body.pop()
            
        self.direction = direction

        # Renderizza il gioco se la modalità è "human".
        if self.render_mode == "human":
            self.graphics.render(self.snake_body, self.apple_location, self.metadata["render_fps"])

        return self._get_obs(), reward, False, False, self._get_info()

    def render(self):
        """ Renderizza l'ambiente. Se la modalità è "rgb_array", restituisce l'immagine renderizzata.
        """
        if self.render_mode == "rgb_array":
            return self.graphics.render(self.snake_body, self.apple_location, self.metadata["render_fps"])

    def close(self):
        """ Chiusura ambiente e rilascio risorse
        """
        self.graphics.close()
        
    def _calculate_distance(self):
        """ Calcola la distanza dalla mela usanso la Distanza Euclidea

        Returns:
            float: Distanza euclidea tra mela e testa del serpente
        """
        return np.sqrt((self.snake_body[0].get_x() - self.apple_location.get_x())**2 + (self.snake_body[0].get_y() - self.apple_location.get_y())**2)
    