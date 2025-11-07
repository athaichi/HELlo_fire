# run_demo.py
import time
from env import FireTractorEnv

def demo_fire_flank(env, fire_start=None, tractor_start=None, max_steps=500, pause=0.4):
    """
    Deterministic fire-flanking demo.
    Tractor follows route, detects fire in observation, backs off, and flanks it.
    """
    obs, info = env.reset(fire_start=fire_start, tractor_start=tractor_start)
    mode = "ROUTE"

    env.render(block=False)
    time.sleep(pause)

    for step in range(max_steps):
        obs, info = env._get_observation(), {}
        action = env._next_route_action(obs=obs)
        obs, done, truncated, info = env.step(action)
        env.render(block=False)
        time.sleep(pause)
        if done:
            break

    print("\n===== FLANK DEMO RESULTS =====")
    for k, v in info.items():
        print(f"{k}: {v}")
    print("==============================\n")


if __name__ == "__main__":
    # Initialize environment
    env = FireTractorEnv(width=50, height=50)

    # Example: fire starts in the middle, tractor starts bottom-left
    demo_fire_flank(
        env,
        fire_start=(25, 25),
        tractor_start=(2, 50, "down"),
        max_steps=800,
        pause=0.25
    )
