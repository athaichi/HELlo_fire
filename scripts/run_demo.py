# run_demo.py
import time
import numpy as np
import pandas as pd
import sys, os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from src
from src.env.env import FireTractorEnv

print("PYTHONPATH:", sys.path)
print("FILES IN PROJECT ROOT:", os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))))

# Import from test
from tests.test_case import TEST_CASES, save_final_png, save_summary_csv

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


def bresenham_line(x0, y0, x1, y1):
    """Generate integer grid cells between (x0, y0) and (x1, y1)."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points

def discretize_route(route_points, width, height):
    """
    Converts float click coordinates into continuous grid-aligned route.
    Returns list of (int x, int y) cells with all interpolated steps.
    """
    # Clamp and round to grid
    route_cells = []
    
    if tractor_start is not None and len(route_points) > 0:
        sx, sy, _ = tractor_start
        x1, y1 = route_points[0]
        sx, sy = int(np.clip(round(sx), 0, width - 1)), int(np.clip(round(sy), 0, height - 1))
        x1, y1 = int(np.clip(round(x1), 0, width - 1)), int(np.clip(round(y1), 0, height - 1))
        route_cells.extend(bresenham_line(sx, sy, x1, y1))


    for i in range(len(route_points) - 1):
        x0, y0 = route_points[i]
        x1, y1 = route_points[i + 1]

        # Snap to grid cell coordinates
        x0, y0 = int(np.clip(round(x0), 0, width - 1)), int(np.clip(round(y0), 0, height - 1))
        x1, y1 = int(np.clip(round(x1), 0, width - 1)), int(np.clip(round(y1), 0, height - 1))

        # Interpolate
        segment = bresenham_line(x0, y0, x1, y1)
        route_cells.extend(segment)

    last_x, last_y = route_points[-1]
    last_x = int(np.clip(round(last_x), 0, width - 1))
    last_y = int(np.clip(round(last_y), 0, height - 1))

    # Compute distances to all four edges
    dist_top = last_y
    dist_bottom = height - 1 - last_y
    dist_left = last_x
    dist_right = width - 1 - last_x

    # Pick nearest edge
    min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
    if min_dist == dist_top:
        edge_point = (last_x, 0)
    elif min_dist == dist_bottom:
        edge_point = (last_x, height - 1)
    elif min_dist == dist_left:
        edge_point = (0, last_y)
    else:
        edge_point = (width - 1, last_y)

    # Connect last route point → edge
    route_cells.extend(bresenham_line(last_x, last_y, *edge_point))

    # Remove duplicates while preserving order
    seen = set()
    final_route = []
    for cell in route_cells:
        if cell not in seen:
            seen.add(cell)
            final_route.append(cell)

    print(f"✅ Discretized route length: {len(final_route)}")
    return final_route

def get_user_route(width, height, fire_start, tractor_start, wind_speed, wind_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title("Click to define route (Press Enter when done)")

    fx, fy = fire_start
    sx, sy, _ = tractor_start

    # Fire + Tractor
    ax.plot(fx, fy, "r*", markersize=12, label="Fire start")
    ax.plot(sx, sy, "go", markersize=8, label="Tractor start")

    # --- WIND INDICATOR ---
    if wind_speed > 0:
        # wind_dir is degrees (0 = east, 90 = north, etc.)
        rad = np.radians(wind_dir)
        dx = np.cos(rad)
        dy = -np.sin(rad)

        ax.quiver(
            width - 10, 5,    # position of arrow
            dx, dy,
            scale=5,
            scale_units="xy",
            color="blue",
            width=0.01,
            label=f"Wind: {wind_speed} m/s, {wind_dir}°"
        )

    ax.legend()
    plt.grid(True)
    plt.pause(0.05)

    print("🖱️ Click to define route points. Press Enter when finished.")
    raw_points = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    route = discretize_route(raw_points, width, height)
    return route


if __name__ == "__main__":
    tractor_start = (25, 0, "right")

    for case in TEST_CASES:
        print("\n==============================")
        print(f"Running test case: {case['name']}")
        print("==============================")

        env = FireTractorEnv(
            width=50,
            height=50,
            burn_duration=3.0,
            sense_radius=4,
            moisture=0.05,
            wind_speed=case["wind_speed"],
            wind_dir=case["wind_dir"],
        )

        fire_start = case["fire_start"]

        # Ask user to draw route
        route = get_user_route(
            env.width,
            env.height,
            fire_start,
            tractor_start,
            wind_speed=case["wind_speed"],
            wind_dir=case["wind_dir"],
        )

        obs, info = env.reset(
            fire_start=fire_start,
            tractor_start=tractor_start,
            route=route
        )

        for step in range(2000):
            action = env._next_route_action(obs)
            obs, done, truncated, info = env.step(action)
            env.render(block=False)
            if done:
                break
            time.sleep(0.05)

        total = env.width * env.height
        burned = int(np.sum(env.grid.burned))
        firebreak = len(env.tractor_path)
        saved = total - burned

        print("\n===== FINAL RESULTS =====")
        print(f"🌱 Saved land:       {saved}/{total}")
        print(f"🔥 Burned land:      {burned}")
        print(f"🟪 Firebreak cells:  {firebreak}")
        print(f"🚜 Tractor exited:   {'YES' if env.tractor_exited else 'NO'}")
        print(f"💀 Tractor destroyed:{'YES' if env.tractor_dead else 'NO'}")
        print("==============================\n")

        # --- SAVE PNG ---
        png_path = save_final_png(env, case["name"])

        # Save CSV
        csv_path = save_summary_csv(
            case,
            saved=saved,
            burned=burned,
            total=total,
            tractor_exited=env.tractor_exited,
            tractor_dead=env.tractor_dead,
        )