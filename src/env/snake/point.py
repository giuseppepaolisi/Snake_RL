class Point:
    """
         Definisce un punto con coordinate x, y.
    """
    def __init__(self, x, y):
        """ Inizializza un punto con le coordinate x, y.

        Args:
            x (int): Asse x.
            y (int): Asse y.
        """
        self.x = x
        self.y = y
        
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_point(self):
        return (self.x, self.y)
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False
    
    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.get_x(), self.y + other.get_y())
        raise TypeError("Addition only supported between two Points.")

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.get_x(), self.y - other.get_y())
        raise TypeError("Subtraction only supported between two Points.")
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __hash__(self):
        return hash((self.x, self.y))
