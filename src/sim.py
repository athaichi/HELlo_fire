""""
from .grid import Grid
from .fire import FireModel
from .tractor import Tractor
from .planner import Planner
from .viz import render

def run():
    grid = Grid(20, 15)
    fire = FireModel()
    tractor = Tractor(5, 5)
    planner = Planner()

    # basic placeholder loop
    for step in range(5):
        fire.step(grid)
        tractor.sense(grid)
        planner.plan(tractor, grid, goal=None)
        tractor.move()
        render(grid)

if __name__ == "__main__":
    run()
"""
import pybullet as p
import pybullet_data
import time

def run():
    # connect to PyBullet in GUI mode
    physicsClient = p.connect(p.GUI)

    # optional: set search path so PyBullet can find plane.urdf
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # load a simple ground plane
    planeId = p.loadURDF("plane.urdf")

    # add a simple cube to represent the tractor
    start_pos = [0, 0, 0.5]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    tractor_id = p.loadURDF("r2d2.urdf", start_pos, start_orientation)  # you can replace with cube.urdf

    # basic simulation loop
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1./240.)  # keep it real-time

    p.disconnect()

if __name__ == "__main__":
    run()

