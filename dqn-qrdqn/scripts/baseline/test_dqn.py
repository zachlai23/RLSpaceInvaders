import gymnasium as gym
import numpy as np
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import DQN
from minatar import Environment
from collections import Counter

class MinAtarLocalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_name="space_invaders"):
        self.game = Environment(env_name, sticky_action_prob=0.1, difficulty_ramping=True)
        self.action_space = spaces.Discrete(self.game.num_actions())
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.game.state_shape(), dtype=np.float32
        )

    def step(self, action):
        reward, terminated = self.game.act(action)
        obs = self.game.state().astype(np.float32)
        truncated = False
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        obs = self.game.state().astype(np.float32)
        return obs, {}

def main():
    root_dir = Path(__file__).resolve().parents[3]
    model_dir = root_dir / "dqn" / "models" / "baseline"
    
    MODEL_NAME = "DQN_Baseline_Tuned_100k"
    MODEL_PATH = model_dir / MODEL_NAME
    
    env = MinAtarLocalEnv("space_invaders")

    print(f"Loading model from: {MODEL_PATH}")
    try:
        # SB3 load expects a string or path-like object
        model = DQN.load(str(MODEL_PATH))
    except Exception as e:
        print(f"Error: Model not found at {MODEL_PATH}.zip")
        print(f"Actual error: {e}")
        return

    N_EPISODES = 20
    scores, lengths = [], []
    action_counts = Counter()

    print(f"\nRunning {N_EPISODES} test episodes...")
    print("-" * 40)

    for i in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            steps += 1
            
        scores.append(score)
        lengths.append(steps)
        print(f"Episode {i+1:02d}: Score = {score:.1f} | Duration = {steps} steps")

    # Final Report
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    total_actions = sum(action_counts.values())
    
    print("\n" + "="*40)
    print(f"      DQN BASELINE PERFORMANCE REPORT      ")
    print("="*40)
    print(f"Average Score:   {avg_score:.2f} +/- {std_score:.2f}")
    print(f"High Score:      {np.max(scores):.1f}")
    print(f"Avg Survival:    {np.mean(lengths):.1f} frames")
    print("-" * 40)
    print("Behavior (Action Distribution):")
    for act in range(6):
        count = action_counts.get(act, 0)
        pct = (count / total_actions) * 100
        print(f"  Action {act}: {pct:.1f}%")
    print("="*40)

if __name__ == '__main__':
    main()