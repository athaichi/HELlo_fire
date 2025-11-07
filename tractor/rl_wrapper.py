# rl_wrapper.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from env import FireTractorEnv


# -------------------------
#  PPO Policy (Option A)
# -------------------------
class LocalAvoidancePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 11 * 11 + 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, 5)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs_tensor, extra_tensor):
        x = obs_tensor.unsqueeze(1).float() / 3.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, extra_tensor], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return policy, value


# -----------------------------------
#  Gymnasium-compatible Environment
# -----------------------------------
class RLTractorWrapper(gym.Env):
    """
    A Gymnasium-compatible wrapper around FireTractorEnv
    for PPO training.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.env = FireTractorEnv(width=50, height=50, sense_radius=5)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "sensor_map": spaces.Box(low=-1, high=3, shape=(11,11), dtype=np.int8),
            "extra": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        })

    # Gymnasium expects reset() → (obs, info)
    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset()
        return self._process_obs(obs), {}

    # Gymnasium expects step() → (obs, reward, done, truncated, info)
    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action=action, dt=1.0)
        reward = self._compute_reward(obs, base_reward, info)
        return self._process_obs(obs), reward, done, truncated, info

    def render(self):
        self.env.render(block=False)

    # --- internal helpers ---
    def _compute_reward(self, obs, base_reward, info):
        smap = obs["sensor_map"]
        fire_near = np.any(smap == 1)
        burned_near = np.any(smap == 2)
        r = base_reward
        if fire_near:   r -= 1.0
        if burned_near: r -= 0.5
        if info["tractor_active"]: r += 0.5
        if info["tractor_exited"]: r += 100.0
        if info["tractor_dead"]:   r -= 100.0
        return float(np.clip(r, -100, 100))

    def _process_obs(self, obs):
        smap = obs["sensor_map"]
        pose = obs["pose"]
        goal = self._goal_vector()
        fire_dist = [self._nearest_fire_dist()]
        extra = np.concatenate([pose, goal, fire_dist]).astype(np.float32)
        return {"sensor_map": smap, "extra": extra}

    def _goal_vector(self):
        if self.env.route_index < len(self.env.route):
            gx, gy = self.env.route[self.env.route_index]
            dx = gx - self.env.tractor.x
            dy = gy - self.env.tractor.y
            d = np.hypot(dx, dy) + 1e-6
            return [dx/d, dy/d]
        return [0.0, 0.0]

    def _nearest_fire_dist(self):
        tx, ty = self.env.tractor.x, self.env.tractor.y
        ys, xs = np.where(self.env.grid.burning)
        if len(xs) == 0:
            return 1.0
        dist = np.min(np.hypot(xs - tx, ys - ty)) / max(self.env.width, self.env.height)
        return float(np.clip(1.0 - dist, 0.0, 1.0))
