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
        if self.direction in ('left', 'a'):
            self.x -= self.speed
        elif self.direction in ('right', 'd'):
            self.x += self.speed
        elif self.direction in ('up', 'w'):
            self.y -= self.speed
        elif self.direction in ('down', 's'):
            self.y += self.speed

        return self.x, self.y
        

