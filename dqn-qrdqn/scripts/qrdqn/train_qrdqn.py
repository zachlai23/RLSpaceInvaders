import csv
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from pathlib import Path
from gymnasium import spaces
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minatar import Environment

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
        self.action_space = spaces.Discrete(self.game.num_actions())
        
        shape = self.game.state_shape()
        self.observation_space = spaces.Box(
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

class CSVLoggingCallback(BaseCallback):
    def __init__(self, csv_path, log_interval=50_000, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.log_interval = log_interval
        self.last_log_step = 0
        self.episode_count = 0

    def _on_training_start(self):
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['frame', 'episode', 'eval_mean', 'eval_std', 'avg_loss'])

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_count += 1

        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self.last_log_step = self.num_timesteps

            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep['r'] for ep in self.model.ep_info_buffer]
                eval_mean = np.mean(rewards)
                eval_std = np.std(rewards)
            else:
                eval_mean = np.nan
                eval_std = np.nan

            avg_loss = self.model.logger.name_to_value.get('train/loss', np.nan)

            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([self.num_timesteps, self.episode_count, eval_mean, eval_std, avg_loss])

        return True


def main():
    root_dir = Path(__file__).resolve().parents[3]
    log_dir = root_dir / "tb_logs"
    model_save_dir = root_dir / "qrdqn" / "models" / "stacked"
    
    model_save_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        raw_env = MinAtarLocalEnv("space_invaders")
        return Monitor(raw_env) 
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4, channels_order='first')

    TIMESTEPS = 1_000_000
    RUN_NAME = "QRDQN_tuned_1mil" 

    policy_kwargs = dict(
        features_extractor_class=MinAtarCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = QRDQN(
        "CnnPolicy",           
        env,
        policy_kwargs=policy_kwargs, 
        learning_rate=5e-4,          
        batch_size=128,               
        buffer_size=100_000,          
        learning_starts=1000,         
        target_update_interval=1000,
        exploration_fraction=0.3,
        verbose=1,
        tensorboard_log=str(log_dir)
    )
    
    csv_path = model_save_dir / f"{RUN_NAME}_log.csv"
    callback = CSVLoggingCallback(csv_path=csv_path, log_interval=50_000)

    print(f"Initializing Frame Stacked QRDQN Model: {RUN_NAME}...")
    print(f"Training for {TIMESTEPS} steps...")
    print(f"Logging CSV to: {csv_path}")

    model.learn(total_timesteps=TIMESTEPS, tb_log_name=RUN_NAME, callback=callback)

    save_path = model_save_dir / RUN_NAME
    model.save(str(save_path))
    print(f"Done! Model saved to {save_path}.zip")

if __name__ == '__main__':
    main()
