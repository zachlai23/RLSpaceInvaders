import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from pathlib import Path
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minatar import Environment
from collections import Counter

# Custom CNN
class MinAtarCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten() 
        )
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# Environment Wrapper
class MinAtarLocalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name="space_invaders"):
        self.game = Environment(env_name, sticky_action_prob=0.1, difficulty_ramping=True)
        self.action_space = gym.spaces.Discrete(self.game.num_actions())
        
        shape = self.game.state_shape()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(shape[2], shape[0], shape[1]), dtype=np.float32
        )

    def step(self, action):
        reward, terminated = self.game.act(action)
        obs = np.transpose(self.game.state(), (2, 0, 1)).astype(np.float32)
        truncated = False
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        obs = np.transpose(self.game.state(), (2, 0, 1)).astype(np.float32)
        return obs, {}

def main():
    root_dir = Path(__file__).resolve().parents[2]
    MODEL_PATH = root_dir / "models" / "qrdqn" / "QRDQN_tuned_1mil"
    
    def make_env():
        return Monitor(MinAtarLocalEnv("space_invaders"))
    
    env = DummyVecEnv([make_env])
    
    env = VecFrameStack(env, n_stack=4, channels_order='first')

    print(f"Loading Model from: {MODEL_PATH}")
    try:
        model = QRDQN.load(str(MODEL_PATH))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    scores, lengths = [], []
    action_counts = Counter()

    # Evaluation Loop
    print("\nRunning 20 QR-DQN test episodes...")
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
        print(f"Episode {i+1:02d}: Score = {score:.1f} | Duration = {steps} steps")

    # Print Results
    total_actions = sum(action_counts.values())
    
    print("\n" + "=" * 45)
    print("      QR-DQN PERFORMANCE REPORT      ")
    print("=" * 45)
    print(f"Average Score: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")
    print(f"High Score:    {np.max(scores):.1f}")
    print(f"Low Score:     {np.min(scores):.1f}")
    print(f"Avg Survival:  {np.mean(lengths):.1f} frames")
    print("-" * 45)
    print("Behavior (Action Distribution):")
    for a in range(6):
        pct = (action_counts[a] / total_actions) * 100 if total_actions > 0 else 0
        print(f"  Action {a}: {pct:>5.1f}%")

if __name__ == '__main__':
    main()
