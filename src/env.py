# env.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

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
        sense_radius=12,             # in cells
        moisture=0.05,
        wind_speed=15.0,
        wind_dir=90.0,
        cell_size=10.0,
    ):
        self.rng = np.random.default_rng(seed)

        self.width = width
        self.height = height
        self.sense_radius = sense_radius

        # Fire / physics
        self.fire = FireModel(
            moisture=moisture,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            cell_size=cell_size,
            burn_duration=burn_duration,
        )

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

        obs = self._get_observation()
        info = self._get_info(done=False)
        return obs, info


    def step(self, action: int, dt: float = 1.0):
        """
        Applies an action (if tractor still active), advances fire, updates states.
        Returns (obs, done, truncated, info).
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

            self.tractor.move(self.width, self.height)
            tx, ty = self.tractor.x, self.tractor.y

            # Leaving farm -> tractor disappears; fire keeps running
            if tx < 0 or ty < 0 or tx >= self.width or ty >= self.height:
                self.tractor_active = False
                self.tractor_exited = True

            else:
                # Tractor dies on burning OR burned
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
            obs = self._get_observation()
            info = self._get_info(done=True)
            truncated = False
            print("Tractor has died. Ending episode.")
            return obs, True, truncated, info

        # 2) Advance fire model by dt
        self.fire.step(self.grid, dt=dt)  # update .burning and advance ignite_time

        # 3) Enforce burned flag
        self._update_burned_flags()

        # 4) Convergence detection (no more burning, etc.)
        done = self._converged()
        
        obs = self._get_observation()
        info = self._get_info(done=done)

        truncated = False
        return obs, done, truncated, info

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

    # -------- Partial Observations: Sensor --------
    def _cell_code_from_world(self, x, y) -> int:
        """Translate world cell to code {0..3} without tractor."""
        if self.grid.burning[y, x]:
            return STATE_BURNING
        if self.grid.burned[y, x]:
            return STATE_BURNED
        if self.grid.fuel_type[y, x] == 0:  # plowed firebreak
            return STATE_FIREBREAK
        return STATE_EMPTY  # fuel present
    

    def _build_sensor_map_world_aligned(self, radius: int = 5) -> np.ndarray:
        """
        World-aligned (North=up) sensor map centered on tractor.
        Values: -1 unknown/outside radius, else {0..3} matching state codes (no tractor).
        """
        size = 2 * radius + 1  # 11 if radius=5
        smap = np.full((size, size), fill_value=-1, dtype=np.int8)

        cx, cy = self.tractor.x, self.tractor.y
        # if tractor inactive or off-map, still return -1 grid
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            return smap

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # circle mask (use Euclidean; switch to Manhattan if you prefer a diamond)
                if dx*dx + dy*dy > radius * radius:
                    continue
                wx, wy = cx + dx, cy + dy
                iy, ix = dy + radius, dx + radius  # map to [0..size-1]
                # outside map -> unknown
                if not (0 <= wx < self.width and 0 <= wy < self.height):
                    smap[iy, ix] = -1
                    continue
                # look up world state, do NOT draw tractor
                smap[iy, ix] = self._cell_code_from_world(wx, wy)

        return smap

    def _get_observation(self):
        """
        Returns:
        - 'sensor_map': (11,11) int8 world-aligned (North=up), center = tractor, radius=5
                        Values: -1=unknown, 0=fuel, 1=burning, 2=burned, 3=firebreak
        - 'pose': (4,) float32 = [x_norm, y_norm, dx, dy]
        """
        sensor_map = self._build_sensor_map_world_aligned(radius=5)

        # Pose (normalized position; heading as unit vector)
        if self.tractor_active:
            x_norm = float(np.clip(self.tractor.x / max(1, self.width - 1), 0, 1))
            y_norm = float(np.clip(self.tractor.y / max(1, self.height - 1), 0, 1))
            dx, dy = self._dir_to_vec(self.tractor.direction)
        else:
            x_norm = y_norm = 0.0
            dx = dy = 0.0

        return {
            "sensor_map": sensor_map,  # int8 in {-1,0,1,2,3}
            "pose": np.array([x_norm, y_norm, dx, dy], dtype=np.float32),
        }


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

    # ------- Helpers for partial observability --------

    def _local_patch(self, dense_map, cx, cy, size):
        """Return size x size crop centered at (cx,cy). Unknown outside = -1."""
        P = size
        half = P // 2
        patch = np.full((P, P), fill_value=-1, dtype=int)
        for j in range(P):
            for i in range(P):
                x = cx + (i - half)
                y = cy + (j - half)
                if 0 <= x < self.width and 0 <= y < self.height:
                    patch[j, i] = dense_map[y, x]
        return patch

    def _dir_to_angle(self, direction: str) -> float:
        if direction in ("up", "w"):    return -np.pi / 2
        if direction in ("down", "s"):  return  np.pi / 2
        if direction in ("left", "a"):  return  np.pi
        if direction in ("right", "d"): return  0.0
        return 0.0

    def _dir_to_vec(self, direction: str):
        ang = self._dir_to_angle(direction)
        return float(np.cos(ang)), float(np.sin(ang))

    # ------------- DEMO MODE -------------
    def demo(self, fire_start=None, tractor_start=None, max_steps=500, pause=0.4):
        import time
        import matplotlib.pyplot as plt

        obs, info = self.reset(fire_start=fire_start, tractor_start=tractor_start)

        # Initial frame
        self.render(block=False)
        time.sleep(pause)

        for _ in range(max_steps):
            # Demo policy – replace with model action if testing RL
            action = 2
            obs, reward, done, truncated, info = self.step(action)

            self.render(block=False)
            time.sleep(pause)

            # If tractor no longer active, switch to fire-only rollout
            if self.tractor_dead or self.tractor_exited:
                print("🚜 Tractor stopped — rolling fire to completion...")
                self.roll_fire_to_completion()
                break

            # If simulation ends normally
            if done:
                break

        # -------- Summary --------
        total = self.width * self.height
        burned = int(np.sum(self.grid.burned))
        firebreak = len(self.tractor_path)
        saved = total - burned

        survived = (self.tractor_exited or (self.tractor_active and not self.tractor_dead))
        print("\n===== DEMO RESULTS =====")
        print(f"🌱 Saved land:       {saved}/{total}")
        print(f"🔥 Burned land:      {burned}")
        print(f"🟪 Firebreak cells:  {firebreak}")
        print(f"🚜 Tractor survived: {'Yes' if survived else 'No'}")
        print(f"⏱️ Duration:         {self.fire.current_time:.1f} min")
        print("========================\n")

    def roll_fire_to_completion(self):
        """
        Continue simulating fire (without tractor movement) until all burning stops.
        Returns number of fire steps simulated.
        """
        steps = 0
        while np.any(self.grid.burning):
            self.fire.step(self.grid, dt=1.0)
            self._update_burned_flags()
            steps += 1

        # Debug print
        print(
            f"[FireRoll] burn_finished={np.sum(self.grid.burned)} | "
            f"ignited={np.isfinite(self.grid.ignite_time).sum()} | steps={steps}"
        )
        return steps
