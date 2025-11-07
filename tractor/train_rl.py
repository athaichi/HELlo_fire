# train_rl.py
import torch
from rl_wrapper import RLTractorWrapper, LocalAvoidancePolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ---------------------------------------
#  Custom Feature Extractor for PPO
# ---------------------------------------
class FireFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=64)
        self.policy = LocalAvoidancePolicy()

    def forward(self, obs):
        smap = obs["sensor_map"]
        extra = obs["extra"]
        policy, value = self.policy(smap, extra)
        # Return combined features (not used directly; PPO uses its heads)
        return torch.cat([policy, value], dim=1)


# ---------------------------------------
#  Main Training Loop
# ---------------------------------------
def train_local_ppo(total_timesteps=5e5):
    def make_env():
        return RLTractorWrapper()
    env = DummyVecEnv([make_env])

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="auto",
    )

    model.learn(total_timesteps=int(total_timesteps))
    model.save("ppo_tractor_avoidance")

    print("✅ Training finished, model saved to ppo_tractor_avoidance.zip")
    return model


if __name__ == "__main__":
    train_local_ppo()
