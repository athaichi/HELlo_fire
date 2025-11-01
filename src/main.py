from env import FireTractorEnv

env = FireTractorEnv(width=30, height=30)
env.demo(fire_start=(10, 10), tractor_start=(10, -1, "down"))

done = False
while not done:
    action = 0  # noop (replace with policy/planner action)
    obs, reward, done, truncated, info = env.step(action, dt=1.0)
    env.render()