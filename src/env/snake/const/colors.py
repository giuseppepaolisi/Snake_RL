import numpy as np

"""
    Colori del gioco.
"""

BODY_COLOR = np.array([0, 128, 0], dtype=np.uint8)  # Verde scuro per il corpo
HEAD_COLOR = np.array([0, 255, 0], dtype=np.uint8)  # Verde chiaro per la testa
FOOD_COLOR = np.array([255, 0, 0], dtype=np.uint8)  # Rosso per il cibo
SPACE_COLOR = np.array([0, 0, 0], dtype=np.uint8)   # Nero per le celle vuote