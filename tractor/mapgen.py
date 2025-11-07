import numpy as np
from scipy.ndimage import gaussian_filter
from grid import Grid

def generate_soybean_farm(
    width=100,
    height=100,
    base_elev=100,
    noise=3,
    smoothness=5,
    slope_strength=1.0,
    fuel_type=1
):
    """
    Generate a soybean farm with smooth rolling elevation and natural fuel variability.

    Adds:
        - fuel_load_map per cell (±20% variation)
        - moisture_map per cell (±10% variation)
    """

    # --- Elevation Map ---
    raw_noise = np.random.randn(height, width) * noise
    elevation = gaussian_filter(raw_noise, sigma=smoothness)
    elevation = base_elev + slope_strength * elevation

    # --- Uniform fuel type (soybean) ---
    fuel = np.full((height, width), fuel_type, dtype=int)

    # --- Natural Variability Added Here ---
    # Fuel load variation ±20%
    fuel_load_map = np.random.uniform(0.8, 1.2, size=(height, width))

    # Moisture variation ±10%
    moisture_map = np.random.uniform(0.9, 1.1, size=(height, width))

    # Build grid
    grid = Grid(width, height)
    grid.elevation = elevation
    grid.fuel_type = fuel
    grid.burning = np.zeros((height, width), dtype=bool)

    # Attach variability maps
    grid.fuel_load_map = fuel_load_map       # multiplier for params["w_0"]
    grid.moisture_map = moisture_map         # multiplier for base moisture

    return grid
