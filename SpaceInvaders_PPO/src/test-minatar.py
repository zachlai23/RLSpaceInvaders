
import gymnasium as gym
from minatar.gym import register_envs  # explicit registration (works even if entry-point auto-loading fails)

register_envs()

env = gym.make("MinAtar/SpaceInvaders-v0")  # v0 = full 6-action set
obs, info = env.reset()
total = 0
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total += reward
    if terminated or truncated:
        obs, info = env.reset()

print("Total reward after 1000 random steps:", total)
env.close()