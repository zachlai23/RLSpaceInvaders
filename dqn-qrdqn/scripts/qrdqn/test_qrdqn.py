import gymnasium as gym
import numpy as np
from pathlib import Path
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from minatar import Environment
from collections import Counter

class MinAtarLocalEnv(gym.Env):
    def __init__(self, env_name="space_invaders"):
        self.game = Environment(env_name, sticky_action_prob=0.1, difficulty_ramping=True)
        self.action_space = gym.spaces.Discrete(self.game.num_actions())
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.game.state_shape(), dtype=np.float32)

    def step(self, action):
        reward, terminated = self.game.act(action)
        return self.game.state().astype(np.float32), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self.game.state().astype(np.float32), {}

def main():
    root_dir = Path(__file__).resolve().parents[2]
    MODEL_PATH = root_dir / "models" / "qrdqn" / "QRDQN_Stacked_1mil"
    
    def make_env():
        return Monitor(MinAtarLocalEnv("space_invaders"))
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    print(f"Loading Model from: {MODEL_PATH}")
    model = QRDQN.load(str(MODEL_PATH))
    
    scores, lengths = [], []
    action_counts = Counter()

    # Evaluation Loop
    for i in range(20):
        obs = env.reset()
        done, score, steps = False, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_val = int(action[0])
            action_counts[action_val] += 1
            
            obs, rewards, dones, _ = env.step(action)
            done, score, steps = dones[0], score + rewards[0], steps + 1
        
        scores.append(score)
        lengths.append(steps)

    # Print Results
    total_actions = sum(action_counts.values())
    
    print(f"Average Score: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")
    print(f"High Score:    {np.max(scores):.1f}")
    print(f"Low Score:     {np.min(scores):.1f}")
    print(f"Avg Survival:  {np.mean(lengths):.1f} frames")
    print("-" * 45)
    print("Behavior (Action Distribution):")
    # Display distribution for all 6 actions
    for a in range(6):
        pct = (action_counts[a] / total_actions) * 100
        print(f"  Action {a}: {pct:>5.1f}%")

if __name__ == '__main__':
    main()