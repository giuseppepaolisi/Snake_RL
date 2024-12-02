from .point import Point
from queue import deque
from typing import List, Deque

class Snake:
    """
        Definisce un serpente all'interno del gioco di Snake.
    """
    def __init__(self, body: Point = Point(1, 1), direction: Point = Point(0,1), length=10):
        """ Inizializza un oggetto Snake.

        Args:
            body (Point, optional): Definisce la coordinata in cui si trova il serpente. Defaults to Point(1, 1).
            direction (Point, optional): Definisce la direzione verso cui il serpente sta andando. Defaults to Point(0,1).
        """
        self.body: Deque[Point] = deque() # Il serpente è rappresentato come una coda di Point.
        self.body.append(body)
        self.direction = direction
        self.grow = False # Indica se il serpente deve crescere.

        start = body
        for i in range(0, length-1):
            opposite=Point(direction.get_x()*-1, direction.get_y()*-1)
            start = Point(start.get_x() + opposite.get_x(), start.get_y() + opposite.get_y())
            self.body.append(start)

    def get_all_body(self) -> Deque[Point]:
        return self.body
    
    def get_body(self) -> List[Point]:
        return list(self.body)[1:]
    
    def get_head(self)-> Point:
        return self.body[0]
    
    def move(self, direction: Point):
        """ Sposta il serpente di un passo nella direction indicata.

        Args:
            direction (Point): Indica in che posizione il serpente deve spostarsi.
        """
        
        # Gestione direzione opposta
        opposite=Point(direction.get_x()*-1, direction.get_y()*-1)
        if opposite == self.direction:
            return
        
        # Aggiorna la posizione del serpente in base alla direzione
        new_head = Point(
            self.body[0].get_x() + direction.get_x(),
            self.body[0].get_y() + direction.get_y()
        )
        self.body.appendleft(new_head)
        
        if(not self.grow): # False
            self.body.pop() # Elimina l'ultimo punto
        self.direction = direction
    
    def set_grow(self, grow=True):
        self.grow = grow