import gymnasium as gym
import numpy as np
from pathlib import Path
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from minatar import Environment

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
    # load and save paths
    root_dir = Path(__file__).resolve().parents[2]
    log_dir = root_dir / "tb_logs"
    model_save_dir = root_dir / "models" / "qrdqn"

    model_save_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        raw_env = MinAtarLocalEnv("space_invaders")
        return Monitor(raw_env)
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    TIMESTEPS=1_000_000
    RUN_NAME = "QRDQN_Stacked_1mil"

    model = QRDQN(
        "MlpPolicy",
        env, 
        verbose=1, 
        tensorboard_log=str(log_dir)
    )

    print(f"Starting QRDQN Training: {RUN_NAME}...")

    # 4. Train model
    model.learn(
        total_timesteps=TIMESTEPS, 
        tb_log_name=RUN_NAME
    )
    
    model.save(str(model_save_dir / RUN_NAME))
    print(f"Success! Model saved to {model_save_dir / RUN_NAME}.zip")

if __name__ == '__main__':
    main()