import gymnasium as gym
import numpy as np
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from minatar import Environment

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
        truncated = False
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        obs = self.game.state().astype(np.float32)
        return obs, {}

def main():
    root_dir = Path(__file__).resolve().parents[3] 
    log_dir = root_dir / "tb_logs"
    model_save_dir = root_dir / "dqn" / "models" / "baseline"
    
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Setup Environment with Monitor for logging
    raw_env = MinAtarLocalEnv("space_invaders")
    env = Monitor(raw_env)

    TIMESTEPS = 100_000
    RUN_NAME = "DQN_Baseline_Tuned_100k" 

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,          
        batch_size=128,               
        buffer_size=100_000,          
        learning_starts=1000,         
        target_update_interval=1000,  
        verbose=1,
        tensorboard_log=str(log_dir)
    )

    print(f"Training {RUN_NAME} for {TIMESTEPS} steps...")
    
    # Train with run name for tensorboard
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=RUN_NAME) 
    
    # Save model
    save_path = model_save_dir / RUN_NAME
    model.save(str(save_path))
    print(f"Done! Model saved to {save_path}.zip")

if __name__ == '__main__':
    main()