# grid.py
import numpy as np

class Grid:
    """
    Internal simulation state container.
    Coordinates are (x, y) in functions; arrays are indexed [y, x].
    """
    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)

        # Static maps
        self.elevation = np.zeros((self.height, self.width), dtype=float)
        self.fuel_type = np.zeros((self.height, self.width), dtype=int)  # 0 = no fuel
        # Optional per-cell variability (multipliers)
        self.fuel_load_map = np.ones((self.height, self.width), dtype=float)
        self.moisture_map  = np.ones((self.height, self.width), dtype=float)

        # Fire state (Option B)
        self.ignite_time = np.full((self.height, self.width), np.inf, dtype=float)
        self.burn_start  = np.full((self.height, self.width), np.inf, dtype=float)
        self.burning     = np.zeros((self.height, self.width), dtype=bool)  # active right now
        self.burned      = np.zeros((self.height, self.width), dtype=bool)  # done burning forever

    @classmethod
    def from_arrays(cls, elevation: np.ndarray, fuel_type: np.ndarray):
        h, w = elevation.shape
        g = cls(w, h)
        g.elevation = elevation.astype(float, copy=True)
        g.fuel_type = fuel_type.astype(int, copy=True)
        return g

    def ignite(self, x: int, y: int, time: float = 0.0):
        """Start fire at (x,y). Sets burning= True and ignite_time/burn_start = time."""
        self.burning[y, x] = True
        self.ignite_time[y, x] = time
        self.burn_start[y, x] = time

    def neighbors_4(self, x: int, y: int):
        """4-connected neighbors within bounds."""
        if x + 1 < self.width:  yield (x + 1, y)
        if x - 1 >= 0:          yield (x - 1, y)
        if y + 1 < self.height: yield (x, y + 1)
        if y - 1 >= 0:          yield (x, y - 1)
