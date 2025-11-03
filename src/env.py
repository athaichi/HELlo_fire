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
        min_dist = 1, # random, set to 60 feet
        max_dist = 5, # random, set to 120 feet
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

        # ---- Tractor start ----
        if tractor_start is None:
            sx = self.rng.integers(0, self.width)
            sy = -1               # actually start at edge of graph 
            direction = "down"
        else:
            sx, sy, direction = tractor_start

        self.tractor = Tractor(start_x=int(sx), start_y=int(sy), direction=direction, speed=1)

        # pick a goal from goal region
        #self.goal = random.choice(self.goal_region)

        # start Dstarlite
        start = (self.tractor.y, self.tractor.x)
        temp_goal = (self.height-1, self.width -1)
        self.dstar = DStarLite(self.grid.width, self.grid.height, start, temp_goal)
        self.dstar.update_vertex(temp_goal)
        self.compute_cost_map()

        # update and compute goal
        self.goal = self._select_goal()
        self.dstar.goal = self.goal
        self.dstar.rhs[self.goal] = 0
        self.dstar.update_vertex(self.goal)

        info = self._get_info(done=False)
        return info

    def compute_cost_map(self):
        fire_mask = self.grid.burning | self.grid.burned
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

        # 2) Update DStarLight
        self.compute_cost_map()
        self.dstar.start = (ty, tx) # start from tractor position
        self.dstar.km += self.dstar.heuristic(self.dstar.last, self.dstar.start)
        self.dstar.last = self.dstar.start
        self.goal = self._select_goal() # update to nearest goal point
        self.dstar.compute_shortest_path()

        # 3) Move tractor (currently unintuitive, but place tractor on next cell)
        if self.tractor_active:
            # decide on next cell for tractor to move to
            # bias toward moving toward goal
            neighbors = self.dstar.get_neighbors(ty, tx)
            valid_neighbors = [(ny,nx) for ny,nx in neighbors if self.dstar.cost[ny,nx] < np.inf]

        if not valid_neighbors:
            next_cell = (self.tractor.y, self.tractor.x)
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
        
            # If we reach the goal, we allow the tractor to leave (goal is on field boundary)
            gx, gy = self.goal
            if tx == gx and ty == gy:
                self.tractor_active = False
                self.tractor_exited = True

            else:
                # 🔥☠️ Tractor dies on burning OR burned
                if self.grid.burning[ty, tx] or self.grid.burned[ty, tx]:
                    self.tractor_active = False
                    self.tractor_dead = True
                    done = True  # end episode immediately

                else:
                    # Make firebreak (unburnable)
                    self._make_firebreak(tx, ty)
                    self.tractor_path.add((ty, tx)) # keep track for color overlays

        # If tractor just died this step, end now (no more fire advance or rendering needed)
        if done:
            info = self._get_info(done=True)
            truncated = False
            return True, truncated, info

        # 3) Enforce burned flag
        self._update_burned_flags()

        # 4) Convergence detection (no more burning, etc.)
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
        # Tractor starts at top, goal is bottom row
        goal_row = self.height - 1
        candidate_cells = [(goal_row, x) for x in range(self.width)]

        # Filter by safe distance buffer
        safe_cells = [
            (y, x) for y, x in candidate_cells
            if self.min_dist <= self.dist_to_fire[y, x] <= self.max_dist
        ]

        if safe_cells:
            # Pick the cell closest to fire
            goal_cell = min(safe_cells, key=lambda c: self.dist_to_fire[c[0], c[1]])
        else:
            # fallback: any cell on the goal row
            goal_cell = random.choice(candidate_cells)

        return goal_cell

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
