from fire import FireModel
from tractor import Tractor
from mapgen import generate_soybean_farm
import matplotlib.pyplot as plt
import numpy as np

def run():
    grid = generate_soybean_farm(width=50, height=50, base_elev=100, noise=3, smoothness=5)
    grid.ignite(0, 40)  # seed in middle

    # inits
    fire = FireModel(moisture=0.05, wind_speed=15.0, wind_dir=90.0, cell_size=10.0)
    tractor = Tractor(start_x = 10, start_y = 20, direction = "down", speed = 1)

    # keep track of tractor path
    tractor_path = set()

    plt.ion()
    fig, ax = plt.subplots()

    for step in range(60):
        # move fire and tractor
        fire.step(grid, dt=1.0)
        tractor.move(grid.width, grid.height)
        tx, ty = tractor.x, tractor.y

        # simple collision logic 
        # 1. stop if tractor runs into burning soybeans
        if grid.burning[ty, tx]: 
            print(f"Tractor ran into fire at ({tractor.x}, {tractor.y})")
            break 
        
        # 2. change plowed fuel type to be fire break
        grid.fuel_type[ty,tx] = 2

        #print(f"Tractor at: ({tractor.x}, {tractor.y})")

        # update tractor path
        tractor_path.add((ty, tx)) # stored as row, column

        ax.clear()

        # Start from terrain background
        ax.imshow(grid.elevation, cmap="terrain", interpolation="nearest")

        # Build state map
        state = np.zeros((grid.height, grid.width), dtype=int)
        state[grid.burning] = 1
        state[(grid.ignite_time < fire.current_time) & (~grid.burning)] = 2
        # add tractor path cells 
        for (y,x) in tractor_path:
            state[y,x] = 3 # set as firebreak color
        # add tractor cell
        state[ty, tx] = 4

        # Custom colormap: 0 transparent, 1 red (burning), 2 black (burned), 3 purple (firebreak), 4 orange (tractor)
        from matplotlib.colors import ListedColormap
        cmap_fire = ListedColormap(["none", "red", "black", "purple", "gold"])

        # Overlay fire states
        ax.imshow(state, cmap=cmap_fire, interpolation="nearest", alpha=0.7)

        ax.set_title(f"Time {fire.current_time:.1f} min")
        plt.pause(0.3)




    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run()
