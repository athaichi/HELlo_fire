# rl_wrapper.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from copy import deepcopy
import random

def preprocess_obs(obs: dict) -> np.ndarray:
    sensor_flat = obs["sensor_map"].astype(np.float32).reshape(-1)  # 11x11 -> 121
    return np.concatenate([sensor_flat, obs["pose"].astype(np.float32)], axis=0)  # 121 + 4 = 125


def _random_border_start(width, height):
    side = random.randint(0, 3)
    if side == 0:     # top
        x, y, d = random.randint(0, width-1), 0, "down"
    elif side == 1:   # bottom
        x, y, d = random.randint(0, width-1), height - 1, "up"
    elif side == 2:   # left
        x, y, d = 0, random.randint(0, height-1), "right"
    else:             # right
        x, y, d = width - 1, random.randint(0, height-1), "left"
    return (x, y, d)


class FireTractorGymWrapper(gym.Env):
    """
    Observation: flat vector of [sensor_map(121), pose(4)]
    Action space: Discrete(5)

    Episode ends when:
      - fire fully finishes (primary terminal), or
      - time cap reached (truncate)

    IMPORTANT: If the tractor exits or dies, simulate the fire to completion
    *inside the same step call*, and credit the final reward to that last tractor step.
    """

    metadata = {"render_modes": []}

    def __init__(self, core_env,
                 max_steps=600,
                 border_start=True,
                 ):
        super().__init__()
        self.core = core_env
        self.max_steps = int(max_steps)
        self.border_start = bool(border_start)
        self._prev_min_fire_dist = None
        self._no_improve_steps = 0

        # Probe obs
        obs, info = self.core.reset()
        vec = preprocess_obs(obs)
        self.obs_shape = vec.shape[0]  # 125

        # REQUIRED by SB3/Gym do not deleted
        self.observation_space = spaces.Box(
            low=np.concatenate([np.full(121, -1.0, dtype=np.float32),
                                np.full(4,   -1.0, dtype=np.float32)]),
            high=np.concatenate([np.full(121,  3.0, dtype=np.float32),
                                np.full(4,    1.0, dtype=np.float32)]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)


        # Episode state
        self._steps = 0                    # counts tractor decision steps ONLY
        self._visited_breaks = set()
        self._breaks_created = 0
        self._exit_step = None             # recorded the moment tractor exits
        self._action_hist = np.zeros(5, dtype=int)

    def reset(self, seed=None, options=None, **kwargs):
        del seed, options
        if self.border_start and ("tractor_start" not in kwargs):
            kwargs["tractor_start"] = _random_border_start(self.core.width, self.core.height)

        obs, info = self.core.reset(**kwargs)

        self._steps = 0
        self._visited_breaks.clear()
        self._breaks_created = 0
        self._exit_step = None
        self._prev_min_fire_dist = None
        self._no_improve_steps = 0


        return preprocess_obs(obs), info


    def step(self, action):
        import numpy as np

        shaped = 0.0

        was_exited = self.core.tractor_exited

        # One env step with the chosen action
        obs2, done_core, truncated_core, info = self.core.step(int(action))
        # Count this as one *decision* step
        self._steps += 1
        
        # Debug
        self._action_hist[int(action)] += 1
        if (self._steps % 100) == 0:
            print(f"[act-hist] steps={self._steps} counts={self._action_hist.tolist()}")
        

        # REWARD new breaks
        total_cells = self.core.width * self.core.height
        if self.core.tractor_active and not self.core.tractor_dead:
            pos = (self.core.tractor.y, self.core.tractor.x)
            if pos in self.core.tractor_path:
                if pos not in self._visited_breaks:
                    shaped += 20 / total_cells
                    self._visited_breaks.add(pos)
                    self._breaks_created += 1
                else:
                    shaped -= 5 / total_cells

        # REWARD + PENALTY: near fire but not too near
        if self.core.tractor_active and not self.core.tractor_dead:
            smap = obs2["sensor_map"]  # 11x11 centered on tractor, north-up
            burning = np.argwhere((smap == 1) | (smap == 2))  # treat burned as dangerous too
            if len(burning) > 0:
                CENTER = 5
                if len(burning) > 0:
                    CENTER = 5
                    dists = np.abs(burning[:,0]-CENTER) + np.abs(burning[:,1]-CENTER)
                    dist = int(dists.min())
                else:
                    dist = None

                # Smooth progress shaping
                if self.core.tractor_active and not self.core.tractor_dead:
                    if dist is not None:
                        if self._prev_min_fire_dist is not None:
                            delta = self._prev_min_fire_dist - dist  # >0 means getting closer (good)
                            if delta > 0:
                                shaped += 0.05 * float(delta)         # reward approach
                                self._no_improve_steps = 0
                            elif delta < 0:
                                shaped -= 0.05 * float(-delta)        # punish drifting away
                                self._no_improve_steps += 1
                            else:
                                self._no_improve_steps += 1
                        else:
                            self._no_improve_steps = 0
                        self._prev_min_fire_dist = dist
                    else:
                        # Can't see danger → encourage exploration, discourage camping
                        shaped -= 0.02
                        self._no_improve_steps += 1

                    # Anti-“straight line / noop” nudge if nothing improved for a while
                    if self._no_improve_steps >= 5:
                        shaped -= 0.05  # tiny boredom penalty


        # Record at which step
        just_exited = (not was_exited) and self.core.tractor_exited
        if just_exited and self._exit_step is None:
            self._exit_step = self._steps  # record tractor's last decision step

        # ---------------- Normal termination checks ----------------
        done = False
        truncated = False

        # EXIT - too many steps: Cap on agent decision steps (truncate; no outcome reward)
        if self._steps >= self.max_steps:
            truncated = True

        # EXIT - tractor died or exist: Continue fire if tractor exited or died
        terminal_due_to_exit_or_death = (self.core.tractor_dead or self.core.tractor_exited)
        if terminal_due_to_exit_or_death:
            self.core.roll_fire_to_completion()
            done = True

        # DONE and fire finished
        if done and not truncated:
            burned = np.sum(self.core.grid.burned)                  # burned
            burning = np.sum(self.core.grid.burning)                # currently burning
            firebreak = np.sum(self.core.grid.fuel_type == 2)       # plowed
            not_saved = burned + burning + firebreak
            final_saved = total_cells - not_saved

            # REWARD: final saved land percentage
            shaped += final_saved / total_cells

            # PENALTY: death penalty
            if self.core.tractor_dead:
                shaped -= 3.0

            # Debug
            if (self._exit_step is not None) and (self._exit_step < 50):
                print(f"⚠️Tractor exited at step {self._exit_step} ")

            # Log outcome
            info["final_saved_land"] = final_saved
            info["breaks_created"] = self._breaks_created
            info["episode_steps"] = self._steps
            info["exit_step"] = self._exit_step if self._exit_step is not None else None

        elif truncated: # i.e. stopped by RL limit, not real stop
            # do NOT add final outcome reward (to avoid bias/instability)
            info["final_saved_land"] = None
            info["breaks_created"] = self._breaks_created
            info["episode_steps"] = self._steps
            info["exit_step"] = self._exit_step if self._exit_step is not None else None

        # Return next obs (post-world-step), shaped reward, flags, info
        return preprocess_obs(obs2), shaped, done, truncated, info

    # Planning helpers
    def clone(self):
        return deepcopy(self)

    def simulate(self, action):
        c = self.clone()
        return c.step(action)
