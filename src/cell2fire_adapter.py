import subprocess, os, glob, pandas as pd

class Cell2FireAdapter:
    def __init__(self, binary="./fire/Cell2Fire", instance=".fire/fire_instance/",
                 outdir="./fire/out", dt_hours=1.0):
        self.binary = binary
        self.instance = instance
        self.outdir = outdir
        self.dt_hours = dt_hours
        os.makedirs(outdir, exist_ok=True)

        # Run once to generate grids
        cmd = [
            self.binary,
            "--input-instance-folder", self.instance,
            "--output-folder", self.outdir,
            "--ignitions",
            "--sim-years", "1", "--nsims", "1",
            "--grids", "--final-grid",
            "--Fire-Period-Length", str(self.dt_hours),
            "--weather", "rows", "--nweathers", "1",
            "--seed", "42"
        ]
        subprocess.run(cmd, check=True)

        # collect all grid CSVs
        self.grid_files = sorted(
            glob.glob(os.path.join(self.outdir, "**", "*.csv"), recursive=True)
        )
        self.t = -1

    def step(self):
        """Return a boolean mask (HxW) for the next timestep."""
        self.t += 1
        if self.t >= len(self.grid_files):
            return None
        df = pd.read_csv(self.grid_files[self.t])
        arr = df.to_numpy()
        # adjust depending on encoding (check first CSV)
        return (arr == 1)
