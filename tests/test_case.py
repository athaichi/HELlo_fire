import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# save_final_png --> help record the final image of the environment
# save_summary_csv --> help record the summary of the test cases
# TEST_CASES defined at the bottom

def save_final_png(env, case_name):
    """Save the final render of the environment as a PNG."""
    filename = case_name.replace(" ", "_") + ".png"
    path = os.path.join(RESULT_DIR, filename)

    env.render(block=False)
    plt.savefig(path, dpi=200)
    plt.close()

    return path


def save_summary_csv(case, saved, burned, total, tractor_exited, tractor_dead):
    """Append or replace a row of summary results in summary.csv."""
    csv_path = os.path.join(RESULT_DIR, "summary.csv")

    record = {
        "case_name": case["name"],
        "fire_start": str(case["fire_start"]),
        "wind_speed": case["wind_speed"],
        "wind_dir": case["wind_dir"],
        "saved_land": saved,
        "burned_land": burned,
        "total_land": total,
        "saved_percent": saved / total,
        "tractor_exited": tractor_exited,
        "tractor_dead": tractor_dead,
    }

    # If exists, load and replace
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Remove any row with same case name
        df = df[df["case_name"] != case["name"]]

        # Append the new record
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    else:
        # Create new file
        df = pd.DataFrame([record])

    # Save back
    df.to_csv(csv_path, index=False)

    return csv_path


TEST_CASES = [
    {
        "name": "Top Right, no wind",
        "fire_start": (40, 10),
        "wind_speed": 10,
        "wind_dir": 0,
    },
    {
        "name": "Middle, no wind",
        "fire_start": (25, 25),
        "wind_speed": 10,
        "wind_dir": 0,
    },
    {
        "name": "Top Left, no wind",
        "fire_start": (10, 10),
        "wind_speed": 10,
        "wind_dir": 0,
    },
    {
        "name": "Middle Top, no wind",
        "fire_start": (25, 10),
        "wind_speed": 10,
        "wind_dir": 0,
    },
    {
        "name": "Top Right, strong wind down",
        "fire_start": (40, 10),
        "wind_speed": 50,
        "wind_dir": 180,      # down (north→south)
    },
    {
        "name": "Bottom Left, strong wind right",
        "fire_start": (10, 10),
        "wind_speed": 50,
        "wind_dir": 90,       # right
    },
]
