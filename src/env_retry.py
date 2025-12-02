# env.py (State-Machine Rewrite)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from enum import Enum
from scipy.ndimage import distance_transform_edt
from heapq import heappush, heappop, heapify

# Project imports (unchanged)
from grid import Grid
from mapgen import generate_soybean_farm
from fire import FireModel
from tractor import Tractor


# --------------------
# CELL STATES (visual only)
# --------------------
STATE_EMPTY     = 0
STATE_BURNING   = 1
STATE_BURNED    = 2
STATE_FIREBREAK = 3
STATE_TRACTOR   = 4


# ===================================================================
#                       D *  L I T E
# ===================================================================
class DStarLite:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.start = start
        self.goal  = goal

        self.g = np.full((height, width), np.inf)
        self.rhs = np.full((height, width), np.inf)
        if goal is not None:
            self.rhs[goal] = 0

        self.U = []
        self.km = 0
        self.cost = np.ones((height, width))
        self.last = start

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def calculate_key(self, s):
        g_s, rhs_s = self.g[s], self.rhs[s]
        k1 = min(g_s, rhs_s) + self.heuristic(self.start, s) + self.km
        k2 = min(g_s, rhs_s)
        return (k1, k2)

    def get_neighbors(self, y, x):
        neighbors = []
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]: 
            ny, nx = y+dy, x+dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                neighbors.append((ny, nx))
        return neighbors

    def update_vertex(self, s):
        y, x = s
        if s != self.goal:
            neigh = self.get_neighbors(y, x)
            self.rhs[y, x] = min([self.cost[ny, nx] + self.g[ny, nx] for ny, nx in neigh])

        # remove existing U entries
        self.U = [(k, v) for (k, v) in self.U if v != s]
        heapify(self.U)

        if self.g[y, x] != self.rhs[y, x]:
            heappush(self.U, (self.calculate_key(s), s))

    def compute_shortest_path(self):
        if self.goal is None:
            return  # safeguard

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


# ===================================================================
#                       TRACTOR STATE MACHINE
# ===================================================================
class TractorState(Enum):
    INIT = 0
    WAYPOINT = 1
    EXIT = 2
    DONE = 3
    DEAD = 4


# ===================================================================
#                       F I R E   T R A C T O R   E N V
# ===================================================================
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
        min_dist=5,
        max_dist=10,
    ):
        self.rng = np.random.default_rng(seed)

        self.width = width
        self.height = height

        self.fire = FireModel(
            moisture=moisture,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            cell_size=cell_size,
            burn_duration=burn_duration,
        )

        # waypoint distance constraints
        self.min_dist = min_dist
        self.max_dist = max_dist

        # For rendering
        self._fig = None
        self._ax = None

    # ================================================================
    #                        RESET
    # ================================================================
    def reset(self, fire_start=None, tractor_start=None):
        self.grid = generate_soybean_farm(width=self.width, height=self.height)
        self.step_idx = 0
        self.current_time = 0.0

        # ---- FIRE START ----
        if fire_start is None:
            ix = self.rng.integers(0, self.width)
            iy = self.rng.integers(0, self.height)
        else:
            ix, iy = fire_start

        self.grid.ignite(ix, iy, time=0.0)
        print(f"ignition point (x,y): {ix},{iy}")
        self.fire_origin = (iy, ix)

        # ---- TRACTOR START ----
        if tractor_start is None:
            sx = self.rng.integers(0, self.width)
            sy = -1
            direction = "down"
        else:
            sx, sy, direction = tractor_start

        self.tractor = Tractor(start_x=int(sx), start_y=int(sy),
                               direction=direction, speed=1)
        print(f"tractor start point (x,y): {sx},{sy}")

        # ---- D* INIT ----
        start = (self.tractor.y, self.tractor.x)
        temp_goal = (self.height - 1, self.width - 1)
        self.dstar = DStarLite(self.width, self.height, start, temp_goal)
        self.dstar.update_vertex(temp_goal)

        # ---- Waypoint tracking ----
        self.visited_waypoints = set()
        self.waypoints = {}

        # ---- Path logs ----
        self.tractor_path = set()

        # ---- State ----
        self.state = TractorState.INIT
        self.goal = None
        self.current_goal_dir = None

        # ---- Bookkeeping for testing ----
        self.tractor_dead = False
        self.tractor_done = False

        # compute cost map for first step
        self.compute_cost_map()
        self._update_waypoints()

        return self._get_info(done=False)

    # ================================================================
    #                      COST MAP
    # ================================================================
    def compute_cost_map(self):
        fire_mask = self.grid.burning
        dist = distance_transform_edt(~fire_mask)

        cost = np.ones((self.height, self.width))
        cost[fire_mask] = np.inf
        cost[dist > self.max_dist] += 5
        cost[dist < self.min_dist] = np.inf

        self.dist_to_fire = dist
        self.dstar.cost = cost

        # update D* rhs for all cells
        for y in range(self.height):
            for x in range(self.width):
                self.dstar.update_vertex((y, x))

    # ================================================================
    #                      WAYPOINT UPDATE
    # ================================================================
    def _update_waypoints(self):
        fire_mask = self.grid.burning | self.grid.burned
        dist = self.dist_to_fire

        safe = np.argwhere(
            (~fire_mask) &
            (dist >= self.min_dist) &
            (dist <= self.max_dist)
        )

        new_wp = {}

        for d in ["N", "S", "W", "E"]:
            if d in self.visited_waypoints:
                continue

            if safe.size == 0:
                continue

            if d == "N":
                idx = np.argmin(safe[:, 0])
            elif d == "S":
                idx = np.argmax(safe[:, 0])
            elif d == "W":
                idx = np.argmin(safe[:, 1])
            else:
                idx = np.argmax(safe[:, 1])

            new_wp[d] = tuple(safe[idx])

        self.waypoints = new_wp
        print("safe ring cells:", safe.shape[0])
        print("waypoints generated:", list(new_wp.keys()))

    # ================================================================
    #                WAYPOINT GOAL SELECTION
    # ================================================================
    def _choose_next_waypoint(self):
        remaining = {k: v for k, v in self.waypoints.items()
                     if k not in self.visited_waypoints}

        if not remaining:
            return None, None

        ty, tx = self.tractor.y, self.tractor.x
        direction, pos = min(
            remaining.items(),
            key=lambda item: abs(item[1][0] - ty) + abs(item[1][1] - tx)
        )
        return pos, direction

    # ================================================================
    #                EXIT GOAL SELECTION
    # ================================================================
    def _find_safe_exit_cells(self, safety_radius):
        safe = []
        burning = np.where(self.grid.burning)
        if len(burning[0]) == 0:
            min_dist = np.inf

        for y in range(self.height):
            for x in range(self.width):
                if y in (0, self.height-1) or x in (0, self.width-1):
                    if not self.grid.burning[y, x] and not self.grid.burned[y, x]:
                        if len(burning[0]) > 0:
                            min_dist = np.min([
                                abs(y - by) + abs(x - bx)
                                for by, bx in zip(*burning)
                            ])
                        if min_dist >= safety_radius:
                            safe.append((y, x))
        return safe

    def _find_reachable_safe_exit(self, safety_radius):
        safe_edges = self._find_safe_exit_cells(safety_radius)
        ty, tx = self.tractor.y, self.tractor.x

        reachable = []
        for ey, ex in safe_edges:
            self.dstar.start = (ty, tx)
            self.dstar.goal = (ey, ex)
            self.dstar.compute_shortest_path()
            if not np.isinf(self.dstar.g[ey, ex]):
                reachable.append((ey, ex))
        return reachable

    def _choose_exit_goal(self):
        safe_edges = self._find_reachable_safe_exit(self.min_dist)
        if not safe_edges:
            return None
        ty, tx = self.tractor.y, self.tractor.x
        return min(safe_edges, key=lambda c: abs(c[0]-ty) + abs(c[1]-tx))

    # ================================================================
    #                       TRACTOR MOVE
    # ================================================================
    def _move_tractor_toward_goal(self):
        print("here")
        ty, tx = self.tractor.y, self.tractor.x
        gy, gx = self.goal
        print(f"self.goal: {gx}, {gy}")

        self.dstar.start = (ty, tx)
        self.dstar.goal = (gy, gx)
        print(f"before: {self.dstar.rhs[gy, gx]}")
        self.dstar.rhs[gy, gx] = 0 # manually reset as startpoint = 0
        self.dstar.update_vertex((gy, gx))
        print(f"after: {self.dstar.rhs[gy, gx]}")
        self.dstar.km += self.dstar.heuristic(self.dstar.last, self.dstar.start)
        self.dstar.last = self.dstar.start
        self.dstar.compute_shortest_path()

        if np.isinf(self.dstar.g[gy, gx]):
            print("false 1")
            return False

        neighbors = self.dstar.get_neighbors(ty, tx)
        valid = [
            (ny, nx)
            for ny, nx in neighbors
            if self.dstar.cost[ny, nx] < np.inf
            and not self.grid.burning[ny, nx]
            and not self.grid.burned[ny, nx]
        ]

        if not valid:
            print("false 2")
            return False

        next_cell = min(valid, key=lambda n: self.dstar.g[n])
        self.tractor.y, self.tractor.x = next_cell
        return True

    # ================================================================
    #                       FIREBREAK
    # ================================================================
    def _make_firebreak(self, x, y):
        self.grid.fuel_type[y, x] = 0
        self.grid.burning[y, x] = False
        self.grid.ignite_time[y, x] = np.inf

    # ================================================================
    #                       STEP (STATE MACHINE)
    # ================================================================
    def step(self, action: int, dt: float = 1.0):
        self.step_idx += 1
        self.current_time += dt

        # ---- FIRE ----
        self.fire.step(self.grid, dt)
        self._update_burned_flags()

        ty, tx = self.tractor.y, self.tractor.x
        if 0 <= ty < self.height and 0 <= tx < self.width:
            if self.grid.burning[ty, tx] or self.grid.burned[ty, tx]:
                self.state = TractorState.DEAD
                self.tractor_dead = True
                return True, False, self._get_info(done=True)

        # ---- COST + WAYPOINTS ----
        self.compute_cost_map()
        self._update_waypoints()

        # ----------------------------
        #       STATE TRANSITIONS
        # ----------------------------
        print(f"current state: {self.state}, tractor position: {self.tractor.x},{self.tractor.y}")
        if self.state == TractorState.INIT:
            if self.tractor.y < 0:
                # Move into the grid safely, bypass D* for this first step
                self.tractor.y = 0
                # Keep x the same
                self.state = TractorState.WAYPOINT
                print("state change: init -> waypoint")
                # Pick the first waypoint
                self.goal, self.current_goal_dir = self._choose_next_waypoint()
                print(f"heading towards: {self.current_goal_dir}")
                if self.goal is None:
                    self.state = TractorState.EXIT
                    print("state change: waypoint -> exit")
                return False, False, self._get_info(done=False)

            else:
                self.goal, self.current_goal_dir = self._choose_next_waypoint()
                if self.goal is None:
                    self.state = TractorState.EXIT
                    print("state change: init -> exit")
                else:
                    # D* Lite needs to know about the new goal
                    self.dstar.goal = self.goal
                    self.dstar.update_vertex(self.goal)
                    self.dstar.compute_shortest_path()
                    self.state = TractorState.WAYPOINT
                    print("state change: init -> waypoint")

        elif self.state == TractorState.WAYPOINT:
            if (ty, tx) == self.goal:
                print("goal reached")
                self.visited_waypoints.add(self.current_goal_dir)
                self.goal, self.current_goal_dir = self._choose_next_waypoint()
                if self.goal is None:
                    print("state change: waypoint -> exit")
                    self.state = TractorState.EXIT

        elif self.state == TractorState.EXIT:
            if self.goal is None:
                exit_goal = self._choose_exit_goal()
                if exit_goal is None:
                    self.state = TractorState.DEAD
                    self.tractor_dead = True
                    return True, False, self._get_info(done=True)
                self.goal = exit_goal

            if (ty, tx) == self.goal:
                self.state = TractorState.DONE
                self.tractor_done = True
                return True, False, self._get_info(done=True)

        elif self.state in (TractorState.DONE, TractorState.DEAD):
            return True, False, self._get_info(done=True)

        # ----------------------------
        #         MOVE TRACTOR
        # ----------------------------
        if self.state in (TractorState.WAYPOINT, TractorState.EXIT):
            print("fallthrough")
            ok = self._move_tractor_toward_goal()
            print(f"ok? {ok}")
            if not ok:
                self.state = TractorState.DEAD
                self.tractor_dead = True
                return True, False, self._get_info(done=True)

            ty, tx = self.tractor.y, self.tractor.x
            self._make_firebreak(tx, ty)
            self.tractor_path.add((ty, tx))

        return False, False, self._get_info(done=False)

    # ================================================================
    #                       BURNING → BURNED
    # ================================================================
    def _update_burned_flags(self):
        now = self.fire.current_time
        ignite = self.grid.ignite_time
        dur = self.fire.burn_duration

        done = (ignite < np.inf) & (now >= ignite + dur)
        self.grid.burned |= done
        self.grid.burning[done] = False

    # ================================================================
    #                       CONVERGENCE
    # ================================================================
    def _converged(self):
        return not np.any(self.grid.burning)

    # ================================================================
    #                       STATE MAP (VISUAL)
    # ================================================================
    def _build_state_map(self):
        H, W = self.grid.height, self.grid.width
        state = np.full((H, W), STATE_EMPTY, dtype=int)

        state[self.grid.burning] = STATE_BURNING
        state[self.grid.burned & (~self.grid.burning)] = STATE_BURNED

        mask = (self.grid.fuel_type == 0) & (~self.grid.burning) & (~self.grid.burned)
        state[mask] = STATE_FIREBREAK

        if self.state not in (TractorState.DEAD, TractorState.DONE):
            tx, ty = self.tractor.x, self.tractor.y
            if 0 <= ty < H and 0 <= tx < W:
                state[ty, tx] = STATE_TRACTOR

        return state

    # ================================================================
    #                       INFO DICT
    # ================================================================
    def _get_info(self, done: bool):
        total = self.width * self.height
        burned = int(np.sum(self.grid.ignite_time < np.inf))
        saved = total - burned
        return {
            "time": float(self.fire.current_time),
            "burning_now": int(np.sum(self.grid.burning)),
            "burned_total": burned,
            "saved_total": saved,
            "tractor_state": self.state.name,
            "done": bool(done),
        }

    # ================================================================
    #                       RENDER
    # ================================================================
    def render(self, block=False):
        state = self._build_state_map()

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots()

        ax = self._ax
        ax.clear()

        ax.imshow(self.grid.elevation, cmap="terrain", interpolation="nearest")

        cmap = ListedColormap(["none", "red", "black", "purple", "gold"])
        ax.imshow(state, cmap=cmap, interpolation="nearest", alpha=0.7)

        # draw waypoints
        for direction, (wy, wx) in self.waypoints.items():
            ax.plot(wx, wy,
                    marker='o',
                    color='cyan' if direction not in self.visited_waypoints else 'green',
                    markersize=6)

        ax.set_title(f"t={self.fire.current_time:.1f} | burning={self.grid.burning.sum()}")
        plt.pause(0.001)
        if block:
            plt.show()

    # ------------- DEMO MODE -------------

    def demo(self, fire_start=None, tractor_start=None, max_steps=500, pause=0):
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
            if self.state == TractorState.DEAD:
                break

            self.render(block=False)
            time.sleep(pause)

            if done:
                break

        # If tractor died, close figure; else hold final frame
        if self.state == TractorState.DEAD:
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

        survived = (self.state == TractorState.DONE or ((self.state == TractorState.WAYPOINT or self.state == TractorState.EXIT) and not (self.state == TractorState.DEAD)))
        print("\n===== DEMO RESULTS =====")
        print(f"🌱 Saved land:       {saved}")
        print(f"🔥 Burned land:      {burned}")
        print(f"🟪 Firebreak cells:  {firebreak}")
        print(f"🚜 Tractor survived: {'Yes' if survived else 'No'}")
        print(f"⏱️ Duration:         {self.fire.current_time:.1f} min")
        print("========================\n")

