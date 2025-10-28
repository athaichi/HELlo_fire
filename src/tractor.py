class Tractor:
    def __init__(self, start_x, start_y, direction, speed):
        self.x = start_x
        self.y = start_y
        self.direction = direction
        self.speed = speed

    def sense(self, grid):
        """Sense local environment."""
        pass

    def move(self, grid_width, grid_height):
        """Decide and apply movement."""
        if self.direction == 'left' or self.direction == 'a': 
            self.x = max(self.x - self.speed, 0)
        elif self.direction == 'right' or self.direction == 'd': 
            self.x = min(self.x + self.speed, grid_width - 1)
        elif self.direction == 'up' or self.direction == 'w':
            self.y = max(self.y - self.speed, 0)
        elif self.direction == 'down' or self.direction == 's':
            self.y = min(self.y + self.speed, grid_height - 1)
        

