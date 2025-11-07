# env.py (clean version)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from mapgen import generate_soybean_farm
from fire import FireModel
from tractor import Tractor

# --------------------
# ACTIONS:
# 0: noop, 1: up, 2: down, 3: left, 4: right
# --------------------
STATE_EMPTY     = 0
STATE_BURNING   = 1
STATE_BURNED    = 2
STATE_FIREBREAK = 3
STATE_TRACTOR   = 4


class FireTractorEnv:
    """Simulation environment for tractor + spreading fire."""

    def __init__(
        self,
        width=50,
        height=50,
        burn_duration=3.0,
        sense_radius=5,
        moisture=0.05,
        wind_speed=15.0,
        wind_dir=90.0,
        cell_size=10.0,
    ):
        self.width = width
        self.height = height
        self.sense_radius = sense_radius

        self.fire = FireModel(
            moisture=moisture,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            cell_size=cell_size,
            burn_duration=burn_duration,
        )

        self.rng = np.random.default_rng()
        self._fig = None
        self._ax = None

    # ------------- Core Loop -------------

    def reset(self, fire_start=None, tractor_start=None, route=None):
        self.grid = generate_soybean_farm(width=self.width, height=self.height)
        self.fire.current_time = 0.0

        # Fire start
        if fire_start is None:
            fx, fy = self.rng.integers(0, self.width), self.rng.integers(0, self.height)
        else:
            fx, fy = fire_start
        self.grid.ignite(fx, fy, time=0.0)

        # Tractor start
        if tractor_start is None:
            sx, sy, direction = 2, self.height - 2, "up"
        else:
            sx, sy, direction = tractor_start

        self._make_firebreak(sx, sy)

        self.tractor = Tractor(start_x=sx, start_y=sy, direction=direction, speed=1)
        self.tractor_active = True
        self.tractor_dead = False
        self.tractor_exited = False
        self.tractor_path = set()
        if route == None:
            self.route = self._make_default_l_route()
        else:
            self.route = route
        self.route_index = 0

        obs = self._get_observation()
        return obs, self._get_info(done=False)

    def step(self, action=None, dt=1.0):
        """Step simulation: move tractor (route-following or manual) + advance fire."""
        done = False
        self._apply_action(action)

        # Check if tractor survived or exited
        tx, ty = self.tractor.x, self.tractor.y
        if tx < 0 or ty < 0 or tx >= self.width or ty >= self.height:
            self.tractor_exited = True
            self.tractor_active = False
            # done = True
        elif self.grid.burning[ty, tx] or self.grid.burned[ty, tx]:
            self.tractor_dead = True
            self.tractor_active = False
            done = True
        else:
            self._make_firebreak(tx, ty)
            self.tractor_path.add((ty, tx))

        # Fire update
        self.fire.step(self.grid, dt=dt)
        self._update_burned_flags()

        if not np.any(self.grid.burning):
            done = True

        info = self._get_info(done)
        return self._get_observation(), done, False, info

    # ------------- Rendering -------------

    def render(self, block=False):
        state = self._build_state_map()

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots()

        ax = self._ax
        ax.clear()

        ax.imshow(np.zeros_like(state), cmap="Greys", interpolation="nearest")
        cmap = ListedColormap(["none", "red", "black", "purple", "gold"])
        ax.imshow(state, cmap=cmap, interpolation="nearest", alpha=0.7)

        if self.route:
            # Draw full route in cyan
            rx, ry = zip(*self.route)
            ax.plot(rx, ry, linestyle='--', linewidth=1.0, color='cyan', alpha=0.7)

            # Highlight current target waypoint
            i = int(self.route_index)
            if 0 <= i < len(self.route):
                gx, gy = self.route[i]
                ax.plot(gx, gy, "yo", markersize=8, label="Current waypoint")  # 🟡 Yellow dot

        ax.set_title(
            f"Mode: {getattr(self.tractor, 'mode', '?')} | "
            f"t={self.fire.current_time:.1f} min | burning={int(self.grid.burning.sum())}"
        )
        ax.set_xticks([]); ax.set_yticks([])
        plt.pause(0.001)
        if block:
            plt.show()

    # ------------- Fire + Route Helpers -------------

    def _make_default_l_route(self):
        vertical = [(2, y) for y in range(self.height - 2, self.height // 3, -1)]
        y_mid = self.height // 3
        horizontal = [(x, y_mid) for x in range(2, self.width - 2)]
        return vertical + horizontal
    
    def _is_uturn(self, i):
        """Check if route[i-1] → route[i] → route[i+1] forms a U-turn (sharp angle)."""
        if i <= 0 or i >= len(self.route) - 1:
            return False
        x0, y0 = self.route[i - 1]
        x1, y1 = self.route[i]
        x2, y2 = self.route[i + 1]
        v1 = np.array([x1 - x0, y1 - y0])
        v2 = np.array([x2 - x1, y2 - y1])
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return False
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        print(f"Angle at route index {i}: {angle_deg:.1f}°")
        return angle_deg > 90  # treat >135° as U-turn


    def _next_route_action(self, obs=None):
        """Always move closer to the next waypoint; never increase distance."""
        if not self.route or not self.tractor_active:
            return 0

        tx, ty = self.tractor.x, self.tractor.y
        i = int(np.clip(self.route_index, 0, len(self.route) - 1))
        gx, gy = self.route[i]

        # --- Advance index when we're basically at waypoint ---
        if np.hypot(gx - tx, gy - ty) < 1.0:
            self.route_index = min(i + 1, len(self.route) - 1)
            gx, gy = self.route[self.route_index]

        # --- Evaluate all possible moves ---
        moves = {
            1: (tx, ty - 1),  # up
            2: (tx, ty + 1),  # down
            3: (tx - 1, ty),  # left
            4: (tx + 1, ty),  # right
        }

        # Compute current distance
        current_dist = np.hypot(gx - tx, gy - ty)

        # Among moves that reduce distance, pick the best one
        best_action, best_dist = None, float("inf")
        for action, (nx, ny) in moves.items():
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            new_dist = np.hypot(gx - nx, gy - ny)
            if new_dist < current_dist and new_dist < best_dist:
                best_action, best_dist = action, new_dist

        if best_action is None:
            # already as close as possible
            return 0

        # --- Optional: bias away from fire but never pick a move that increases distance ---
        if obs is not None:
            smap = obs["sensor_map"]
            fire_mask = (smap == 1) | (smap == 2)
            if np.any(fire_mask):
                c = smap.shape[0] // 2
                top_fire = np.sum(fire_mask[:c, :])
                bottom_fire = np.sum(fire_mask[c+1:, :])
                left_fire = np.sum(fire_mask[:, :c])
                right_fire = np.sum(fire_mask[:, c+1:])

                # Adjust preference if multiple "closer" options exist
                if best_action in (4, 3):  # moving horizontally
                    if top_fire > bottom_fire * 1.3:
                        return 2 if 2 in moves else best_action  # prefer down
                    elif bottom_fire > top_fire * 1.3:
                        return 1 if 1 in moves else best_action
                elif best_action in (1, 2):  # moving vertically
                    if left_fire > right_fire * 1.3:
                        return 4 if 4 in moves else best_action
                    elif right_fire > left_fire * 1.3:
                        return 3 if 3 in moves else best_action

        return best_action


    def _apply_action(self, action):
        if action == 1:
            self.tractor.direction = "up"
        elif action == 2:
            self.tractor.direction = "down"
        elif action == 3:
            self.tractor.direction = "left"
        elif action == 4:
            self.tractor.direction = "right"
        self.tractor.move(self.width, self.height)

    def _make_firebreak(self, x, y):
        self.grid.fuel_type[y, x] = 0
        self.grid.burning[y, x] = False
        self.grid.ignite_time[y, x] = np.inf

    def _update_burned_flags(self):
        now = self.fire.current_time
        ignite = self.grid.ignite_time
        dur = self.fire.burn_duration
        done_burning = (ignite < np.inf) & (now >= ignite + dur)
        self.grid.burned |= done_burning
        self.grid.burned[self.grid.burning] = False

    # ------------- Observation + Info -------------

    def _build_state_map(self):
        H, W = self.grid.height, self.grid.width
        state = np.full((H, W), STATE_EMPTY, dtype=int)
        state[self.grid.burning] = STATE_BURNING
        state[self.grid.burned & (~self.grid.burning)] = STATE_BURNED
        firebreak_mask = (self.grid.fuel_type == 0) & (~self.grid.burning) & (~self.grid.burned)
        state[firebreak_mask] = STATE_FIREBREAK
        if self.tractor_active:
            tx, ty = self.tractor.x, self.tractor.y
            if 0 <= tx < self.width and 0 <= ty < self.height:
                state[ty, tx] = STATE_TRACTOR
        return state

    def _build_sensor_map_world_aligned(self):
        radius = self.sense_radius
        size = 2 * radius + 1
        smap = np.full((size, size), -1, dtype=np.int8)
        if not self.tractor_active:
            return smap

        cx, cy = self.tractor.x, self.tractor.y
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                wx, wy = cx + dx, cy + dy
                iy, ix = dy + radius, dx + radius
                if 0 <= wx < self.width and 0 <= wy < self.height:
                    smap[iy, ix] = self._cell_code_from_world(wx, wy)
        return smap

    def _cell_code_from_world(self, x, y):
        if self.grid.burning[y, x]:
            return STATE_BURNING
        if self.grid.burned[y, x]:
            return STATE_BURNED
        if self.grid.fuel_type[y, x] == 0:
            return STATE_FIREBREAK
        return STATE_EMPTY

    def _get_observation(self):
        sensor_map = self._build_sensor_map_world_aligned()
        if self.tractor_active:
            x_norm = self.tractor.x / (self.width - 1)
            y_norm = self.tractor.y / (self.height - 1)
        else:
            x_norm = y_norm = 0.0
        return {"sensor_map": sensor_map, "pose": np.array([x_norm, y_norm], dtype=np.float32)}

    def _get_info(self, done):
        total = self.width * self.height
        ignited = int(np.sum(self.grid.ignite_time < np.inf))
        saved = total - ignited
        return {
            "time": float(self.fire.current_time),
            "burning_now": int(np.sum(self.grid.burning)),
            "burned_total": int(np.sum(self.grid.burned)),
            "saved_total": saved,
            "tractor_active": bool(self.tractor_active),
            "tractor_dead": bool(self.tractor_dead),
            "tractor_exited": bool(self.tractor_exited),
            "done": bool(done),
        }
