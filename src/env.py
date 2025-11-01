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
        obs_patch_size=11,
        rays_front=16,
        rays_side=8,
        sense_radius=12,             # in cells
        ray_stride=1,                # step in cells along ray
        moisture=0.05,
        wind_speed=15.0,
        wind_dir=90.0,
        cell_size=10.0,
    ):
        assert obs_patch_size % 2 == 1, "obs_patch_size must be odd"
        self.rng = np.random.default_rng(seed)

        self.width = width
        self.height = height
        self.obs_patch_size = obs_patch_size
        self.rays_front = rays_front
        self.rays_side = rays_side
        self.sense_radius = sense_radius
        self.ray_stride = max(1, ray_stride)

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
            reward = 0.0
            obs = self._get_observation()
            info = self._get_info(done=True)
            truncated = False
            return obs, reward, True, truncated, info

        # 2) Advance fire model by dt
        self.fire.step(self.grid, dt=dt)  # update .burning and advance ignite_time

        # 3) Enforce burned flag
        self._update_burned_flags()

        # 4) Convergence detection (no more burning, etc.)
        done = self._converged()
        

        # 5) Reward: + cells saved delta on this step (optional placeholder)
        reward = 0.0  # TODO for RL; D* won’t use it.

        obs = self._get_observation()
        info = self._get_info(done=done)

        truncated = False
        return obs, reward, done, truncated, info

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

    # -------- Partial Observations: Sensor --------

    def _get_observation(self):
        """
        Returns a dict with:
          - 'local_grid': (P x P) patch centered at tractor (integers: 0..3; -1 unknown outside map)
          - 'sensor_front': (rays_front,) distances normalized to [0,1]
          - 'sensor_side':  (rays_side,)  distances normalized to [0,1]
          - 'pose': (x_norm, y_norm, dx, dy)  (simple heading as unit vector)
        Both RL & D*
        """
        # Build a latent state map for local patch using visualization coding (without tractor):
        dense = self._build_state_map()
        dense[dense == STATE_TRACTOR] = STATE_EMPTY  # don’t leak tractor marker

        # Local patch
        patch = self._local_patch(dense, self.tractor.x, self.tractor.y, self.obs_patch_size)

        # Ray sensors
        front = self._ray_scan(angle_center=self._dir_to_angle(self.tractor.direction), n_rays=self.rays_front)
        side  = self._ray_scan(angle_center=self._dir_to_angle(self.tractor.direction) + np.pi/2,
                               n_rays=self.rays_side)

        # Pose (normalized to [0,1] for pos; direction as unit vector)
        # TODO: handle if we are placing sensors in various places
        x_norm = np.clip(self.tractor.x / max(1, self.width - 1), 0, 1) if self.tractor_active else 0.0
        y_norm = np.clip(self.tractor.y / max(1, self.height - 1), 0, 1) if self.tractor_active else 0.0
        dx, dy = self._dir_to_vec(self.tractor.direction) if self.tractor_active else (0.0, 0.0)

        return {
            "local_grid": patch.astype(np.int8),          # shape (P,P); values in {-1,0,1,2,3}
            "sensor_front": front.astype(np.float32),     # shape (rays_front,), 0..1
            "sensor_side":  side.astype(np.float32),      # shape (rays_side,),   0..1
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

    def _ray_scan(self, angle_center, n_rays):
        """
        Cast n_rays over 180° FOV centered at angle_center.
        Return distances to nearest 'blocking' or 'fire' feature normalized by sense_radius.
        Blocking here = boundary or burned/firebreak; you can tweak what rays see.
        """
        dists = np.full(n_rays, self.sense_radius, dtype=float)
        half = np.pi / 2  # 180° total
        angles = np.linspace(angle_center - half, angle_center + half, n_rays)

        for i, ang in enumerate(angles):
            for step in range(1, self.sense_radius + 1, self.ray_stride):
                x = int(round(self.tractor.x + step * np.cos(ang)))
                y = int(round(self.tractor.y + step * np.sin(ang)))
                # Out of bounds -> hit
                if not (0 <= x < self.width and 0 <= y < self.height):
                    dists[i] = step
                    break
                # Decide what counts as a "hit":
                # e.g., burning OR burned OR firebreak (obstacle to future spread/tractor)
                if self.grid.burning[y, x] or self.grid.burned[y, x] or (self.grid.fuel_type[y, x] == 0):
                    dists[i] = step
                    break

        # Normalize to [0,1]
        dists = np.clip(dists / max(1, self.sense_radius), 0.0, 1.0)
        return dists

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
            action = 2  # straight down
            obs, reward, done, truncated, info = self.step(action)

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
