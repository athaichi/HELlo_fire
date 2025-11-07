# run_demo.py
import time
from env import FireTractorEnv
import matplotlib.pyplot as plt
import numpy as np

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

def get_user_route(width, height, fire_start, tractor_start):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title("Click to define route (Press Enter when done)")

    fx, fy = fire_start
    sx, sy, _ = tractor_start
    ax.plot(fx, fy, "r*", markersize=12, label="Fire start")
    ax.plot(sx, sy, "go", markersize=8, label="Tractor start")
    ax.legend()
    plt.grid(True)
    plt.pause(0.1)

    print("🖱️ Click to define route points. Press Enter when finished.")
    raw_points = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    # 🔢 Discretize route to grid
    route = discretize_route(raw_points, width, height)
    return route


if __name__ == "__main__":
    # Bigger world
    env = FireTractorEnv(width=80, height=80)

    # Random fire start each run (but shown to user)
    fx = np.random.randint(env.width // 4, 3 * env.width // 4)
    fy = np.random.randint(env.height // 4, 3 * env.height // 4)
    fire_start = (fx, fy)

    # Tractor starts just outside left border, pointing right
    tractor_start = (0, env.height // 2, "right")

    # Let user draw a route, snapped to grid & connected from start
    route = get_user_route(env.width, env.height, fire_start, tractor_start)

    # Reset env with that route
    obs, info = env.reset(fire_start=fire_start, tractor_start=tractor_start, route=route)

    # Main sim loop: tractor follows route; fire always spreads; stops when fire out
    for step in range(2000):
        action = env._next_route_action(obs=obs)
        obs, done, truncated, info = env.step(action)
        env.render(block=False)
        if done:
            break

    # ---- Final summary ----
    total = env.width * env.height
    burned = int(np.sum(env.grid.burned))
    firebreak = len(env.tractor_path)
    saved = total - burned

    print("\n===== FINAL SIMULATION RESULTS =====")
    print(f"🌱 Saved land:       {saved}/{total}")
    print(f"🔥 Burned land:      {burned}")
    print(f"🟪 Firebreak cells:  {firebreak}")
    print(f"🚜 Tractor exited:   {'✅' if env.tractor_exited else '❌'}")
    print(f"💀 Tractor destroyed:{'✅' if env.tractor_dead else '❌'}")
    print(f"⏱️ Duration:         {env.fire.current_time:.1f} min")
    print("==============================\n")
