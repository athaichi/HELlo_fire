# env.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from scipy.ndimage import distance_transform_edt
from heapq import heappush, heappop

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

        # Grid information
        self.burning = np.zeros((height, width), dtype=bool)
        self.burned  = np.zeros((height, width), dtype=bool)
        self.fuel    = np.ones((height, width), dtype=bool)

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
            sy = 0
            direction = "down"
        else:
            sx, sy, direction = tractor_start

        self.tractor = Tractor(start_x=int(sx), start_y=int(sy), direction=direction, speed=1)

        info = self._get_info(done=False)
        return info


    def step(self, action: int, dt: float = 1.0):
        """
        Applies an action (if tractor still active), advances fire, updates states.
        Returns (obs, reward, done, truncated, info).
        """
        self.step_idx += 1
        self.current_time += dt

        # 1) Tractor move (only if active)
        tx, ty = self.tractor.x, self.tractor.y
        done = False  # <-- track early termination

        if self.tractor_active:
            if action == 1:   self.tractor.direction = "up"
            elif action == 2: self.tractor.direction = "down"
            elif action == 3: self.tractor.direction = "left"
            elif action == 4: self.tractor.direction = "right"
            # 0 = noop

            self.tractor.move(self.width, self.height)  # must NOT clamp inside
            tx, ty = self.tractor.x, self.tractor.y

            # Leaving farm -> tractor disappears; fire keeps running
            if tx < 0 or ty < 0 or tx >= self.width or ty >= self.height:
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
                    self.tractor_path.add((ty, tx))

        # If tractor just died this step, end now (no more fire advance or rendering needed)
        if done:
            info = self._get_info(done=True)
            truncated = False
            return True, truncated, info

        # 2) Advance fire model by dt
        self.fire.step(self.grid, dt=dt)  # update .burning and advance ignite_time

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
