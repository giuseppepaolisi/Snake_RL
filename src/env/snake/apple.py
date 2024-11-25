from .point import Point
class Apple:
    """
        Definisce una mela all'interno del gioco di Snake.
    """
    
    def __init__(self, position:Point):
        """ Inizializza un oggetto Apple.

        Args:
            position (Point): Posizione della mela nello spazio tramite coordinate (x, y)
        """
        self.position: Point = position
    
    def get_position(self) -> Point:
        """ Ritorna la posizione della mela.

        Returns:
            Point: Ritorna un oggetto Point con le coordinate della mela.
        """
        if isinstance(self.position, Point):
            return self.position
        