import os
import numpy as np
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from rl_wrapper import FireTractorGymWrapper
from env import FireTractorEnv

# ---------- Curriculum spec ----------
STAGES = [
    #{"size": 5,   "steps": 50_000, "max_steps": 100,  "min_break_before_exit": 1},
    #{"size": 10,  "steps": 100_000, "max_steps": 200,  "min_break_before_exit": 2},
    #{"size": 20,  "steps": 200_000, "max_steps": 400,  "min_break_before_exit": 3},
    {"size": 50,  "steps": 400_000, "max_steps": 800,  "min_break_before_exit": 4},
    #{"size": 100, "steps": 600_000, "max_steps": 1200, "min_break_before_exit": 5},
]
# Same fire across stages: set once here
FIRE_KW = dict(
    burn_duration=3.0,
    moisture=0.05,
    wind_speed=15.0,
    wind_dir=90.0,
    cell_size=10.0,
)

SAVE_DIR = "./curriculum_models"

def make_env(size: int, max_steps: int, seed: int):
    core = FireTractorEnv(width=size, height=size,
                          seed=seed, **FIRE_KW)
    # Always start on border; exit allowed, but only if min_break_before_exit satisfied
    env = FireTractorGymWrapper(
        core_env=core,
        max_steps=max_steps,
        border_start=True,
    )
    return env

def train_stage(stage_idx: int, prev_model_path: str | None):
    cfg = STAGES[stage_idx]
    size   = cfg["size"]
    steps  = cfg["steps"]
    max_s  = cfg["max_steps"]
    min_br = cfg["min_break_before_exit"]

    print(f"\n===== STAGE {stage_idx} | size={size} | steps={steps} =====")

    env = make_env(size=size, max_steps=max_s,  seed=stage_idx*101 + 1)
    eval_env = make_env(size=size, max_steps=max_s, seed=stage_idx*101 + 7)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_dir = os.path.join(SAVE_DIR, f"stage{stage_idx}_best")
    os.makedirs(best_dir, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        eval_freq=10_000,
        deterministic=True,
        n_eval_episodes=5,
        verbose=1,
    )

    if prev_model_path and os.path.exists(prev_model_path):
        print(f"Loading previous model: {prev_model_path}")
        model = RecurrentPPO.load(prev_model_path, env=env, print_system_info=True)
        model.learning_rate = 3e-4
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,            # keep long rollouts for LSTM
            batch_size=1024,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.02,           # ↑ encourage exploration
            vf_coef=0.7,             # stronger value loss to stabilize
            max_grad_norm=0.5,
            verbose=1,
        )


    model.learn(total_timesteps=steps, callback=eval_cb)
    final_path = os.path.join(SAVE_DIR, f"stage{stage_idx}_final.zip")
    model.save(final_path)
    print(f"Saved stage {stage_idx} model -> {final_path}")

    # Prefer best checkpoint if present; else final
    best_zip = os.path.join(best_dir, "best_model.zip")
    next_input = best_zip if os.path.exists(best_zip) else final_path
    return next_input

if __name__ == "__main__":
    next_model = None  # CONTINUE from scratch at stage 0, then promote
    for s in range(len(STAGES)):
        next_model = train_stage(s, prev_model_path=next_model)

    print("\n✅ Curriculum training complete.")
