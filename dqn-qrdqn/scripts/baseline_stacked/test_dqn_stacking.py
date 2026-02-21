import gymnasium as gym
import numpy as np
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from minatar import Environment
from collections import Counter

class MinAtarLocalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name="space_invaders"):
        self.game = Environment(env_name, sticky_action_prob=0.1, difficulty_ramping=True)
        self.action_space = spaces.Discrete(self.game.num_actions())
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.game.state_shape(), dtype=np.float32
        )

    def step(self, action):
        reward, terminated = self.game.act(action)
        obs = self.game.state().astype(np.float32)
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self.game.state().astype(np.float32), {}

def main():
    root_dir = Path(__file__).resolve().parents[3]
    model_dir = root_dir / "dqn" / "models" / "stacked"
    
    MODEL_NAME = "DQN_Stacked_Tuned_100k"
    MODEL_PATH = model_dir / MODEL_NAME

    # Frame stacked environment (must match training setup)
    def make_env():
        return MinAtarLocalEnv("space_invaders")
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    
    print(f"Loading stacked model from: {MODEL_PATH}")
    try:
        model = DQN.load(str(MODEL_PATH))
    except Exception as e:
        print(f"Error: Model not found at {MODEL_PATH}.zip")
        print(f"Actual error: {e}")
        return

    N_EPISODES = 20
    scores, lengths = [], []
    action_counts = Counter()

    print(f"\nRunning {N_EPISODES} stacked test episodes...")
    print("-" * 45)

    for i in range(N_EPISODES):
        # VecEnv reset returns only the observation array, not the info dict
        obs = env.reset() 
        done = False
        score = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # VecEnv returns an array of actions, so we take the first one: action[0]
            action_val = int(action[0])
            action_counts[action_val] += 1
            
            # VecEnv step returns arrays for rewards and dones
            obs, rewards, dones, infos = env.step(action)
            
            done = dones[0]
            score += rewards[0]
            steps += 1
            
        scores.append(score)
        lengths.append(steps)
        print(f"Episode {i+1:02d}: Score = {score:.1f} | Duration = {steps} steps")

    # Final report
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    total_actions = sum(action_counts.values())

    print("\n" + "="*45)
    print(f"      DQN STACKED PERFORMANCE REPORT      ")
    print("="*45)
    print(f"Model Tested:    {MODEL_NAME}")
    print(f"Average Score:   {avg_score:.2f} +/- {std_score:.2f}")
    print(f"High Score:      {np.max(scores):.1f}")
    print(f"Low Score:       {np.min(scores):.1f}")
    print(f"Avg Survival:    {np.mean(lengths):.1f} frames")
    print("-" * 45)
    print("Behavior (Action Distribution):")
    # Show distribution for all 6 possible actions
    for act in range(6):
        count = action_counts.get(act, 0)
        pct = (count / total_actions) * 100
        print(f"  Action {act}: {pct:>5.1f}%")
    print("="*45)

if __name__ == '__main__':
    main()