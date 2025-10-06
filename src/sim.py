from fire import FireModel
from mapgen import generate_soybean_farm
import matplotlib.pyplot as plt
import numpy as np

def run():
    grid = generate_soybean_farm(width=50, height=50, base_elev=100, noise=3, smoothness=5)
    grid.ignite(0, 40)  # seed in middle

    fire = FireModel(moisture=0.05, wind_speed=15.0, wind_dir=90.0, cell_size=10.0)

    plt.ion()
    fig, ax = plt.subplots()

    for step in range(40):
        fire.step(grid, dt=1.0)

        ax.clear()

        # Start from terrain background
        ax.imshow(grid.elevation, cmap="terrain", interpolation="nearest")

        # Build state map
        state = np.zeros((grid.height, grid.width), dtype=int)
        state[grid.burning] = 1
        state[(grid.ignite_time < fire.current_time) & (~grid.burning)] = 2
        print(f"\nStep {step}, time={fire.current_time:.1f}")
        print(state)

        # Custom colormap: 0 transparent, 1 red (burning), 2 black (burned)
        from matplotlib.colors import ListedColormap
        cmap_fire = ListedColormap(["none", "red", "black"])

        # Overlay fire states
        ax.imshow(state, cmap=cmap_fire, interpolation="nearest", alpha=0.7)

        ax.set_title(f"Time {fire.current_time:.1f} min")
        plt.pause(0.3)




    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run()
