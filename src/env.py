# env.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from scipy.ndimage import distance_transform_edt
from heapq import heappush, heappop, heapify

from grid import Grid
from mapgen import generate_soybean_farm
from fire import FireModel
from tractor import Tractor

# --------------------
# ACTIONS (discrete):
# 0: noop
# 1: up
# 2: down
# 3: left
# 4: right
# --------------------

STATE_EMPTY     = 0  # fuel present
STATE_BURNING   = 1  # currently burning
STATE_BURNED    = 2  # burned out
STATE_FIREBREAK = 3  # tractor-plowed, no fuel, unburnable
STATE_TRACTOR   = 4  # visualization only

class DStarLite:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.start = start
        self.goal  = goal
        self.g = np.full((height, width), np.inf)
        self.rhs = np.full((height, width), np.inf)
        self.rhs[goal] = 0
        self.U = []
        self.km = 0
        self.cost = np.ones((height, width))
        self.last = start

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])  # Manhattan calculation

    def calculate_key(self, s):
        g_s, rhs_s = self.g[s], self.rhs[s]
        k1 = min(g_s, rhs_s) + self.heuristic(self.start, s) + self.km
        k2 = min(g_s, rhs_s)
        return (k1, k2)

    def get_neighbors(self, y,x):
        neighbors = []
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]: 
            ny, nx = y+dy, x+dx
            if 0<=nx<self.width and 0<=ny<self.height: #allow us to leave the field
                neighbors.append((ny,nx))
        return neighbors

    def update_vertex(self, s):
        y,x = s
        if s != self.goal:
            self.rhs[y,x] = min([self.cost[ny,nx] + self.g[ny,nx] for ny,nx in self.get_neighbors(y,x)])
        if any(item[1] == s for item in self.U):
            self.U = [(k,v) for k,v in self.U if v != s]
            heapify(self.U)
        if self.g[y,x] != self.rhs[y,x]:
            heappush(self.U, (self.calculate_key(s), s))

    def compute_shortest_path(self):
        while self.U:
            k_old, u = heappop(self.U)
            k_new = self.calculate_key(u)
            if k_old < k_new:
                heappush(self.U, (k_new, u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for n in self.get_neighbors(*u):
                    self.update_vertex(n)
            else:
                g_old = self.g[u]
                self.g[u] = np.inf
                self.update_vertex(u)
                for n in self.get_neighbors(*u):
                    self.update_vertex(n)


class FireTractorEnv:
    def __init__(
        self,
        width=50,
        height=50,
        burn_duration=3.0,
        seed=None,
        moisture=0.05,
        wind_speed=15.0,
        wind_dir=90.0,
        cell_size=10.0,
        min_dist = 5, # random, set to 60 feet
        max_dist = 10, # random, set to 120 feet
    ):
        self.rng = np.random.default_rng(seed)

        self.width = width
        self.height = height
        
        # Fire / physics
        self.fire = FireModel(
            moisture=moisture,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            cell_size=cell_size,
            burn_duration=burn_duration,
        )

        # DStarLite info
        self.goal_region = [(x, self.height - 1) for x in range(self.width)]
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.waypoints = []
        self.visited_waypoints = set()

        # Bookkeeping
        self.current_time = 0.0
        self.step_idx = 0
        self.tractor_active = True
        self.tractor_dead = False
        self.tractor_exited = False
        self.tractor_path = set()  # cells plowed (firebreak)

        # Rendering
        self._fig = None
        self._ax = None

    # ------------- Public API -------------

    def reset(
        self,
        fire_start=None,         # (x, y) or None
        tractor_start=None,      # (x, y, dir) or None
    ):
        self.grid = generate_soybean_farm(width=self.width, height=self.height)
        self.current_time = 0.0
        self.step_idx = 0
        self.tractor_active = True
        self.tractor_dead = False
        self.tractor_exited = False  
        self.tractor_path = set()

        # ---- Fire start ----
        if fire_start is None:
            ix = self.rng.integers(0, self.width)
            iy = self.rng.integers(0, self.height)
        else:
            ix, iy = fire_start
        self.grid.ignite(ix, iy, time=0.0)

        self.fire_origin = (iy, ix)

        # ---- Tractor start ----
        if tractor_start is None:
            sx = self.rng.integers(0, self.width)
            sy = -1               # actually start at edge of graph 
            direction = "down"
        else:
            sx, sy, direction = tractor_start

        self.tractor = Tractor(start_x=int(sx), start_y=int(sy), direction=direction, speed=1)

        # start Dstarlite
        start = (self.tractor.y, self.tractor.x)
        temp_goal = (self.height-1, self.width -1)

        self.dstar = DStarLite(self.grid.width, self.grid.height, start, temp_goal)
        self.dstar.update_vertex(temp_goal)
        self.compute_cost_map()

        # start waypoints
        self._update_waypoints()

        # update and compute goal
        self.goal, self.current_goal_dir = self._select_goal()
        self.dstar.goal = self.goal
        self.dstar.rhs[self.goal] = 0
        self.dstar.update_vertex(self.goal)

        info = self._get_info(done=False)
        return info

    def compute_cost_map(self):
        fire_mask = self.grid.burning 
        dist_to_fire = distance_transform_edt(~fire_mask)
        cost = np.ones((self.height, self.width))
        cost[fire_mask] = np.inf # don't touch the fire
        cost[dist_to_fire > self.max_dist] += 5  # penalty for too far, change as needed
        cost[dist_to_fire < self.min_dist] = np.inf # penalty for too close, change as needed
        self.dstar.cost = cost
        self.dist_to_fire = dist_to_fire
        for y in range(self.height):
            for x in range(self.width):
                self.dstar.update_vertex((y,x))
    
    def _update_waypoints(self):
        fire_mask = self.grid.burning | self.grid.burned
        dist_to_fire = self.dist_to_fire

        new_waypoints = {}
        visited = getattr(self, "visited_waypoints", set())

        for dir_name in ["N", "S", "W", "E"]:
            if dir_name in visited:
                continue

            # Find all safe cells in grid
            safe_cells = np.argwhere(
                (~fire_mask) & (dist_to_fire >= self.min_dist) & (dist_to_fire <= self.max_dist)
            )

            if safe_cells.size == 0:
                print(f"⚠️ Waypoint {dir_name} lost — no safe cell remaining.")
                continue

            # Pick a preferred cell based on direction
            # N -> smallest y, S -> largest y, W -> smallest x, E -> largest x
            if dir_name == "N":
                idx = np.argmin(safe_cells[:, 0])
            elif dir_name == "S":
                idx = np.argmax(safe_cells[:, 0])
            elif dir_name == "W":
                idx = np.argmin(safe_cells[:, 1])
            else:  # E
                idx = np.argmax(safe_cells[:, 1])

            new_waypoints[dir_name] = tuple(safe_cells[idx])

        # make sure to update selected goal waypoint
        if hasattr(self, 'current_goal_dir') and self.current_goal_dir in new_waypoints:
            self.goal = new_waypoints[self.current_goal_dir]
        self.waypoints = new_waypoints

    def step(self, action: int, dt: float = 1.0):
        """
        Applies an action (if tractor still active), advances fire, updates states.
        Returns (done, truncated, info).
        """
        self.step_idx += 1
        self.current_time += dt

        tx, ty = self.tractor.x, self.tractor.y
        done = False  # <-- track early termination

        # 1) Step fire
        self.fire.step(self.grid, dt=dt)  # update .burning and advance ignite_time

        # 2) Update distances and goal
        self.compute_cost_map()
        self._update_waypoints()
        if not hasattr(self, 'visited_waypoints'):
            self.visited_waypoints = set()

        # update goal if needed
        if (not hasattr(self, "goal") or self.goal is None
            or self.grid.burning[self.goal[0], self.goal[1]]
            or self.grid.burned[self.goal[0], self.goal[1]]):
            # Select next unvisited safe waypoint
            self.goal = None
            for dir_name, (wy, wx) in self.waypoints.items():
                if dir_name not in self.visited_waypoints:
                    if not self.grid.burning[wy, wx] and not self.grid.burned[wy, wx]:
                        self.goal = (wy, wx)
                        self.current_goal_dir = dir_name
                        break

        print(self.current_goal_dir)
        # If no safe waypoint left, stop the tractor <- FIX THIS (both success and failure position)
        if self.goal is None:
            self.tractor_active = False
            done = True
            info = self._get_info(done=True)
            return done, False, info

        # 3) Update Dstar
        self.dstar.start = (ty, tx) # start from tractor position
        self.dstar.km += self.dstar.heuristic(self.dstar.last, self.dstar.start)
        self.dstar.last = self.dstar.start
        self.dstar.goal = self.goal
        self.dstar.compute_shortest_path()

        # 4) Move tractor (currently unintuitive, but place tractor on next cell)
        if self.tractor_active:
            # decide on next cell for tractor to move to
            # bias toward moving toward goal
            neighbors = self.dstar.get_neighbors(ty, tx)
            valid_neighbors = [(ny,nx) for ny,nx in neighbors if self.dstar.cost[ny,nx] < np.inf
                               and not self.grid.burning[ny,nx] and not self.grid.burned[ny,nx]]

        if not valid_neighbors:
            next_cell = (self.tractor.y, self.tractor.x) # trapped tractor
        else:
            goal_y, goal_x = self.goal
            # compute combined score: D* cost + safety cost + small forward bias
            def score(n):
                ny, nx = n
                g_val = self.dstar.g[ny, nx]
                cost_val = self.dstar.cost[ny, nx]
                # forward bias: Manhattan distance to goal (smaller is better)
                forward_bias = abs(goal_y - ny) + abs(goal_x - nx)
                return g_val + cost_val + 0.01 * forward_bias  # 0.01 is a small weight
            next_cell = min(valid_neighbors, key=score)

            # move to that cell
            self.tractor.y, self.tractor.x = next_cell
            tx, ty = self.tractor.x, self.tractor.y
        
            # 5) Check if reached waypoint
            if (ty, tx) == self.goal:
                self.visited_waypoints.add(self.current_goal_dir)
                print(f"✅ Waypoint {self.current_goal_dir} reached!")
                self.goal = None # Force next goal selection on step

            # 6) 🔥☠️ Tractor dies on burning OR burned
            if self.grid.burning[ty, tx] or self.grid.burned[ty, tx]:
                self.tractor_active = False
                self.tractor_dead = True
                done = True  # end episode immediately

            else:
                # Make firebreak (unburnable)
                self._make_firebreak(tx, ty)
                self.tractor_path.add((ty, tx)) # keep track for color overlays
            
            # 7) Check mission status
            if all(k in self.visited_waypoints for k in self.waypoints.keys()):
                self.tractor_active = False
                self.tractor_exited = True
                print("🎯 All waypoints reached! Mission complete.")

        # If tractor just died this step, end now (no more fire advance or rendering needed)
        if done:
            info = self._get_info(done=True)
            truncated = False
            return True, truncated, info
        
        # 8) Enforce burned flag
        self._update_burned_flags()

        # 9) Convergence detection (no more burning, etc.)
        if not done: 
            done = self._converged()

        info = self._get_info(done=done)
        truncated = False
        return done, truncated, info

    def render(self, block=False):
        """Matplotlib rendering; safe to call each step."""
        state = self._build_state_map()
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots()

        ax = self._ax
        ax.clear()

        # Terrain
        ax.imshow(self.grid.elevation, cmap="terrain", interpolation="nearest")

        # Overlay states
        cmap = ListedColormap(["none", "red", "black", "purple", "gold"])
        ax.imshow(state, cmap=cmap, interpolation="nearest", alpha=0.7)

        # Overlay waypoints
        for direction, (wy, wx) in self.waypoints.items():
            ax.plot(wx, wy, marker='o', color='cyan' if direction not in self.visited_waypoints else 'green', markersize=6)

        # Title
        ax.set_title(f"t = {self.fire.current_time:.1f} min  |  burning={int(self.grid.burning.sum())}")
        plt.pause(0.001)
        if block: plt.show()

    # ------------- Internal helpers -------------

    def _make_firebreak(self, x, y):
        """Set a cell to permanent firebreak: unburnable and not counted as burned."""
        self.grid.fuel_type[y, x] = 0
        self.grid.burning[y, x] = False
        # keep burned False (it's soil), and prevent future ignition:
        self.grid.ignite_time[y, x] = np.inf

    def _update_burned_flags(self):
        """
        A cell becomes burned only AFTER it has completed burning.
        That is: current_time >= ignite_time + burn_duration.
        Not a very good method... but simple. :(
        """
        now = self.fire.current_time
        ignite = self.grid.ignite_time
        dur = self.fire.burn_duration

        # Finished burning if ignition happened AND burn duration elapsed
        done_burning = (ignite < np.inf) & (now >= ignite + dur)

        # Mark these as burned forever
        self.grid.burned |= done_burning

        # Ensure currently burning cells are never marked as burned
        self.grid.burned[self.grid.burning] = False

    # TODO for RL
    def _converged(self):
        """Fire is done if no cell is burning and no cell ignited this step."""
        # No active burning:
        if np.any(self.grid.burning): 
            return False
        # Additionally, check last-step changes: simplest proxy—if nothing is burning now
        # and FireModel won't schedule any future ignitions (ignite_time not decreasing),
        # we consider it converged. As a safe heuristic: if burning==0 for two consecutive
        # steps, you can track a counter. Here we do the simpler rule:
        return True

    def _build_state_map(self):
        """Dense integer state map for visualization only."""
        H, W = self.grid.height, self.grid.width
        state = np.full((H, W), STATE_EMPTY, dtype=int)

        state[self.grid.burning] = STATE_BURNING
        state[self.grid.burned & (~self.grid.burning)] = STATE_BURNED

        # Firebreaks are fuel_type==0 but not burned
        firebreak_mask = (self.grid.fuel_type == 0) & (~self.grid.burning) & (~self.grid.burned)
        state[firebreak_mask] = STATE_FIREBREAK

        # Tractor position (if inside & active)
        if self.tractor_active:
            tx, ty = self.tractor.x, self.tractor.y
            if 0 <= tx < self.width and 0 <= ty < self.height:
                state[ty, tx] = STATE_TRACTOR

        return state
    
    def _select_goal(self):
        """
        Pick the next active waypoint that is still safe and not yet visited.
        """
        # Recompute dynamic waypoints each step
        self._update_waypoints()

        # Filter out visited waypoints
        remaining = {k: v for k, v in self.waypoints.items() if k not in self.visited_waypoints}
        if not remaining:
            return None, None  # mission complete

        # Pick the closest remaining waypoint (Manhattan distance)
        ty, tx = self.tractor.y, self.tractor.x
        direction, pos = min(remaining.items(), key=lambda item: abs(item[1][0]-ty) + abs(item[1][1]-tx))
        return pos, direction

    # -------- Summary info calculations --------

    def _get_info(self, done: bool):
        total = self.width * self.height
        burned = int(np.sum(self.grid.ignite_time < np.inf))
        saved  = total - burned  # firebreak counts as saved
        return {
            "time": float(self.fire.current_time),
            "burning_now": int(np.sum(self.grid.burning)),
            "burned_total": burned,
            "saved_total": saved,
            "tractor_active": bool(self.tractor_active),
            "done": bool(done),
        }

    # ------------- DEMO MODE -------------

    def demo(self, fire_start=None, tractor_start=None, max_steps=500, pause=0.4):
        import time
        import matplotlib.pyplot as plt

        info = self.reset(fire_start=fire_start, tractor_start=tractor_start)

        # Initial frame
        self.render(block=False)
        time.sleep(pause)

        for _ in range(max_steps):
            action = 2  # straight down
            done, truncated, info = self.step(action)

            # If tractor died, stop immediately (no more rendering)
            if self.tractor_dead:
                break

            self.render(block=False)
            time.sleep(pause)

            if done:
                break

        # If tractor died, close figure; else hold final frame
        if self.tractor_dead:
            if self._fig is not None:
                plt.close(self._fig)
                self._fig, self._ax = None, None
        else:
            self.render(block=True)

        # -------- Summary --------
        total = self.width * self.height
        burned = int(np.sum(self.grid.burned))  # use burned mask
        firebreak = len(self.tractor_path)
        saved = total - burned  # firebreak counts as saved

        survived = (self.tractor_exited or (self.tractor_active and not self.tractor_dead))
        print("\n===== DEMO RESULTS =====")
        print(f"🌱 Saved land:       {saved}")
        print(f"🔥 Burned land:      {burned}")
        print(f"🟪 Firebreak cells:  {firebreak}")
        print(f"🚜 Tractor survived: {'Yes' if survived else 'No'}")
        print(f"⏱️ Duration:         {self.fire.current_time:.1f} min")
        print("========================\n")
