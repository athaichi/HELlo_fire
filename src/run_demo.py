from sb3_contrib.ppo_recurrent import RecurrentPPO
from rl_wrapper import FireTractorGymWrapper
from env import FireTractorEnv
import numpy as np, time

# ======== Stage Config MUST Match Training ========
SIZE = 50
MAX_STEPS = 100
MIN_BREAK = 1
MODEL_PATH = "./curriculum_models/stage0_best/best_model.zip"
# ==================================================

core = FireTractorEnv(width=SIZE, height=SIZE, seed=999)
env = FireTractorGymWrapper(core,
                            max_steps=MAX_STEPS,
                            border_start=True)

model = RecurrentPPO.load(MODEL_PATH, env=env)

fire_xy = (10, 5)
tractor_xy_dir = (6, 48, "down")

obs, info = env.reset(fire_start=fire_xy, tractor_start=tractor_xy_dir)

lstm_state = None
episode_start = np.ones((1,), dtype=bool)
done = truncated = False
final_reward = 0.0
print("\n🚜 Demo Starting\n")
core.render(); time.sleep(0.4)

step = 0
tractor_phase = True

while not (done or truncated):
    if core.tractor_active:  # agent still controls tractor
        action, lstm_state = model.predict(
            obs, state=lstm_state,
            episode_start=episode_start,
            deterministic=True
        )
        obs, reward, done, truncated, info = env.step(int(action))
        print(f"Step {step}: 🚜 Action={action}, Reward={reward:.3f}, Saved={info['saved_total']}")
    final_reward += reward
    core.render()
    time.sleep(0.3)

    episode_start = done or truncated
    step += 1

# Final outcome
total = SIZE * SIZE
burned = np.sum(core.grid.burned)                  # burned
burning = np.sum(core.grid.burning)                # currently burning
firebreak = np.sum(core.grid.fuel_type == 2)       # plowed
not_saved = burned + burning + firebreak
firebreak = len(core.tractor_path)
saved = total - not_saved

print("\n===== FINAL STAGE 0 RESULTS =====")
print(f"Reward:            {info.get('final_reward', 0.0):.3f}")
print(f"🌱 Saved land:       {saved}/{total}")
print(f"🔥 Burned land:      {burned}")
print(f"🟪 Firebreak cells:  {firebreak}")
print(f"🚜 Tractor exited:   {core.tractor_exited}")
print(f"💀 Tractor dead:     {core.tractor_dead}")
print(f"⏱️ Fire spread for {fire_steps} extra steps after exit")
print("=================================\n")
