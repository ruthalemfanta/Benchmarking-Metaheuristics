import gymnasium as gym
import numpy as np
import time

env = gym.make("MountainCarContinuous-v0", render_mode="human")
state, _ = env.reset()

for _ in range(300):
    env.render()
    action = np.array([1.0])
    state, _, _, _, _ = env.step(action)
    time.sleep(0.02)

env.close()
