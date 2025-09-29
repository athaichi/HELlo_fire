from cell2fire_adapter import Cell2FireAdapter

class FireModel:
    def __init__(self):
        self.engine = Cell2FireAdapter(
            binary="./fire/Cell2Fire",
            instance="./fire/fire_instance/",
            outdir="./fire/out",
            dt_hours=1.0
        )

    def step(self, grid):
        mask = self.engine.step()
        if mask is None:
            return
        # map the Cell2Fire mask into your Grid
        grid.on_fire = mask
