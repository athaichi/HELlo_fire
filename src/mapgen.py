import numpy as np
from scipy.ndimage import gaussian_filter
from grid import Grid

def generate_soybean_farm(
    width=100,
    height=100,
    base_elev=100,
    noise=3,
    smoothness=5,
    slope_strength=1.0,   # new knob
    fuel_type=1
):
    """
    Generate a soybean farm with smooth rolling elevation.
    
    Args:
        width, height: farm size in cells
        base_elev: base elevation value
        noise: amplitude of raw elevation variation
        smoothness: Gaussian sigma (higher = smoother hills)
        slope_strength: scale factor for how steep the slopes feel
        fuel_type: integer code for soybean fuel
    """
    # Generate smooth elevation with Gaussian-filtered noise
    raw_noise = np.random.randn(height, width) * noise
    elevation = gaussian_filter(raw_noise, sigma=smoothness)

    # Scale slope effect
    elevation = base_elev + slope_strength * elevation

    # Uniform soybean fuel map
    fuel = np.full((height, width), fuel_type, dtype=int)

    # Build grid
    grid = Grid(width, height)
    grid.elevation = elevation
    grid.fuel_type = fuel
    grid.burning = np.zeros((height, width), dtype=bool)

    return grid
