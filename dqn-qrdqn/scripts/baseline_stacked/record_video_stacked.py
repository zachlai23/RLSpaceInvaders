import gymnasium as gym
import numpy as np
import imageio
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from minatar import Environment
from pathlib import Path

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
        return self.game.state().astype(np.float32), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self.game.state().astype(np.float32), {}

def get_colored_frame(state, scale=30):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    colors = [[0, 255, 0], [255, 0, 0], [100, 0, 0], [0, 0, 255], [255, 255, 255], [255, 255, 0]]
    for channel in range(6): 
        if channel < state.shape[2]:
            mask = state[:, :, channel]
            for c in range(3):
                img[:, :, c] = np.maximum(img[:, :, c], mask * colors[channel][c])
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

def main():
    MODEL_NAME = "DQN_Stacked_100k"

    # Load and Save paths
    root_dir = Path(__file__).resolve().parents[2]
    model_load_path = root_dir / "models" / "baseline_stacked" / MODEL_NAME
    video_save_dir = root_dir / "videos" / "baseline_stacked"
    
    # Ensure video directory exists
    video_save_dir.mkdir(parents=True, exist_ok=True)

    make_env = lambda: MinAtarLocalEnv("space_invaders")
    env = DummyVecEnv([make_env])
    
    # Frame Stacking
    env = VecFrameStack(env, n_stack=4)
    
    print(f"Loading stacked model '{MODEL_NAME}' from {model_load_path}...")
    try:
        model = DQN.load(str(model_load_path))
    except Exception as e:
        print(f"Error: Could not find '{model_load_path}.zip'. Error details: {e}")
        return

    obs = env.reset()
    done = False
    frames = []
    score = 0
    
    print("Recording Stacked Game 1...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        done = dones[0]
        score += rewards[0]
        
        # raw un-stacked grid for video
        current_grid = env.venv.envs[0].game.state()
        frames.append(get_colored_frame(current_grid))

    # Save video
    filename = f"{MODEL_NAME}_score_{int(score)}.mp4"
    save_path = video_save_dir / filename
    
    print(f"Game Over! Score: {score}. Saving {len(frames)} frames to {save_path}...")
    imageio.mimsave(str(save_path), frames, fps=30)
    print("Video saved successfully!")

if __name__ == '__main__':
    main()