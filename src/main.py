from env import FireTractorEnv

env = FireTractorEnv(width=30, height=30)
env.demo(fire_start=(10, 10), tractor_start=(10, -1, "down"))
