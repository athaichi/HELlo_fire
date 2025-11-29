# run.py
import numpy as np
import matplotlib.pyplot as plt
from env import FireTractorEnv
# Import from test
from test import TEST_CASES, save_final_png, save_summary_csv

RESULT_DIR = "results"
#os.makedirs(RESULT_DIR, exist_ok=True)

MAX_STEPS = 2000
PAUSE = 0

def run_test_case(case, render=False):
    """
    Run a simulation based on a test case dict:
      { name, fire_start, wind_speed, wind_dir }

    Collect summary data and save to summary.csv.
    """
    tractor_start = (25, -1, "right")

    env = FireTractorEnv(
        width=50,
        height=50,
        burn_duration=3.0,
        moisture=0.05,
        wind_speed=case["wind_speed"],
        wind_dir=case["wind_dir"],
    )

    import time
    import matplotlib.pyplot as plt

    info = env.reset(fire_start=case["fire_start"], tractor_start=tractor_start)

    # Initial render
    if render:
        env.render(block=False)
        time.sleep(PAUSE)

    # Main loop (same as demo)
    for _ in range(MAX_STEPS):
        action = 2  # always straight down
        done, truncated, info = env.step(action)

        if env.tractor_dead:
            break

        if render:
            env.render(block=False)
            time.sleep(PAUSE)

        if done:
            break

    # Final render
    if render:
        if env.tractor_dead:
            if env._fig is not None:
                plt.close(env._fig)
                env._fig, env._ax = None, None
        else:
            env.render(block=True)

    
    # Compute summary results
   
    total = env.width * env.height
    burned = int(np.sum(env.grid.burned))
    firebreak = len(env.tractor_path)
    saved = total - burned

    #survived = (env.tractor_exited or (env.tractor_active and not env.tractor_dead))

    # For your CSV fields
    tractor_exited = env.tractor_exited
    tractor_dead   = env.tractor_dead

    # --- SAVE PNG ---
    png_path = save_final_png(env, case["name"])

    # Save CSV
    csv_path = save_summary_csv(
        case,
        saved=saved,
        burned=burned,
        total=total,
        tractor_exited=env.tractor_exited,
        tractor_dead=env.tractor_dead,
    )

    # Print summary 
    print(f"Finished: {case['name']}")
    print("\n===== TEST CASE RESULTS =====")
    print(f"Case:              {case['name']}")
    print(f"🌱 Saved land:     {saved}")
    print(f"🔥 Burned land:    {burned}")
    print(f"🟪 Firebreak:      {firebreak}")
    print(f"🚜 Tractor exited: {tractor_exited}")
    print(f"💀 Tractor dead:   {tractor_dead}")
    print(f"⏱️ Duration:       {env.fire.current_time:.1f} min")
    print("=============================\n")

    return env   # or return summary dict if you prefer

def run_all_tests():
    for case in TEST_CASES:
        print(f"Running: {case['name']}")
        run_test_case(case, render=True)

run_all_tests()