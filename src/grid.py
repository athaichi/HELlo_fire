import numpy as np

class Grid:
    def __init__(self, width=None, height=None):
        if width and height:
            self.width = width
            self.height = height
            self.elevation = np.zeros((height, width))
            self.fuel_type = np.zeros((height, width), dtype=int)
            self.burning = np.zeros((height, width), dtype=bool)
            self.burned = np.zeros((height, width), dtype=bool)

    @classmethod
    def from_csv(cls, elev_file, fuel_file):
        elevation = np.loadtxt(elev_file, delimiter=",")
        fuel_type = np.loadtxt(fuel_file, delimiter=",", dtype=int)

        h, w = elevation.shape
        grid = cls(w, h)
        grid.elevation = elevation
        grid.fuel_type = fuel_type
        grid.burning = np.zeros_like(fuel_type, dtype=bool)
        return grid

    def ignite(self, x, y):
        self.burning[y, x] = True

    def neighbors(self, x, y):
        offsets = [(1,0), (-1,0), (0,1), (0,-1)]
        for dx, dy in offsets:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield nx, ny
