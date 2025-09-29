import matplotlib.pyplot as plt

def render(grid):
    """Render the grid (placeholder)."""
    plt.imshow(grid.cells if grid.cells is not None else [[0]])
    plt.show()
