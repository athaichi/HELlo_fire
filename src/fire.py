import numpy as np
from spread import compute_rate_of_spread
from fuel_params import FUEL_MODELS

class FireModel:
    def __init__(self, moisture=0.1, wind_speed=5.0, wind_dir=0.0, 
                 cell_size=10.0, burn_duration=3.0):
        self.moisture = moisture
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir
        self.cell_size = cell_size
        self.burn_duration = burn_duration  # minutes a cell burns
        self.current_time = 0.0  # minutes

    def init_ignite_times(self, grid):
        # Track ignition time and burn start
        grid.ignite_time = np.full((grid.height, grid.width), np.inf)
        grid.burn_start = np.full((grid.height, grid.width), np.inf)

        # Any already-burning cell starts at time 0
        seeds = np.argwhere(grid.burning)
        for (y, x) in seeds:
            grid.ignite_time[y, x] = 0.0
            grid.burn_start[y, x] = 0.0

    def step(self, grid, dt=1.0):
        """
        Advance fire simulation by dt minutes.
        """
        if not hasattr(grid, "ignite_time"):
            self.init_ignite_times(grid)

        self.current_time += dt

        # --- 1. Update burning status ---
        grid.burning = (self.current_time >= grid.ignite_time) & \
                       (self.current_time < grid.ignite_time + self.burn_duration)

        # --- 2. Spread fire from currently burning cells ---
        burning_cells = np.argwhere(grid.burning)
        for (y, x) in burning_cells:
            fuel_type = grid.fuel_type[y, x]
            if fuel_type == 0:
                continue  # no fuel

            params = FUEL_MODELS[fuel_type]

            for nx, ny in grid.neighbors_4(x, y):
                if grid.fuel_type[ny, nx] == 0:
                    continue

                dz = grid.elevation[ny, nx] - grid.elevation[y, x]
                dx, dy = nx - x, ny - y
                dist = np.sqrt(dx**2 + dy**2) * self.cell_size
                slope_mag = np.array([dz / dist if dist > 0 else 0.0])
                slope_dir = np.array([np.arctan2(dy, dx)])

                R = compute_rate_of_spread(
                    np.array([x]), np.array([y]),
                    np.array([nx]), np.array([ny]),
                    np.array([params["w_0"]]),
                    np.array([params["delta"]]),
                    np.array([params["M_x"]]),
                    np.array([params["sigma"]]),
                    np.array([params["h"]]),
                    np.array([params["S_T"]]),
                    np.array([params["S_e"]]),
                    np.array([params["p_p"]]),
                    np.array([self.moisture]),
                    np.array([self.wind_speed]),
                    np.array([self.wind_dir]),
                    slope_mag, slope_dir
                )

                """
                noise = np.random.normal(loc=1.0, scale=0.1)  # 20% variation
                noise = np.clip(noise, 0.1, 1.5)              # prevent negative or crazy values

                R = R * noise
                """
                if R[0] > 0:
                    spread_time = dist / R[0]
                    new_time = self.current_time + spread_time

                    # Update if sooner ignition
                    if new_time < grid.ignite_time[ny, nx]:
                        grid.ignite_time[ny, nx] = new_time
                        grid.burn_start[ny, nx] = new_time
