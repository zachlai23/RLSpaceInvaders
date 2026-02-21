import imageio
import numpy as np
from pathlib import Path
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from minatar import Environment
import gymnasium as gym

# Environment class
class MinAtarLocalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

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

def get_colored_frame(state):
    """Converts a MinAtar 10x10xn binary state into a 100x100 RGB image."""
    colors = [
        [0, 255, 255],   # Cyan
        [255, 0, 255],   # Magenta
        [255, 255, 0],   # Yellow
        [0, 255, 0],     # Green
        [0, 0, 255],     # Blue
        [255, 0, 0]      # Red
    ]
    
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(state.shape[2]):
        mask = state[:, :, i] == 1
        img[mask] = colors[i % len(colors)]
        
    # Upscale 10x10 to 100x100 for better visibility
    return np.repeat(np.repeat(img, 10, axis=0), 10, axis=1)

def main():
    MODEL_NAME = "QRDQN_Stacked_1mil"
    
    # Load and save paths
    root_dir = Path(__file__).resolve().parents[2]
    model_load_path = root_dir / "models" / "qrdqn" / MODEL_NAME
    video_save_dir = root_dir / "videos" / "qrdqn"
    
    video_save_dir.mkdir(parents=True, exist_ok=True)
    
    def make_env(): 
        return MinAtarLocalEnv("space_invaders")
        
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    
    print(f"Loading QRDQN model from: {model_load_path}...")
    try:
        model = QRDQN.load(str(model_load_path))
    except Exception as e:
        print(f"Error: Could not find '{model_load_path}.zip'. Error details: {e}")
        return
        
    obs = env.reset()
    done, frames, score = False, [], 0
    
    print(f"Recording Game for {MODEL_NAME}...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(action)
        
        # Unpack vectorized rewards and dones
        done = dones[0] 
        score += rewards[0]
        
        # Get the underlying 10x10 state for the frame
        current_grid = env.venv.envs[0].game.state()
        frames.append(get_colored_frame(current_grid))

    filename = video_save_dir / f"{MODEL_NAME}_Score_{int(score)}.mp4"
    imageio.mimsave(str(filename), frames, fps=15)
    print(f"Video saved successfully to: {filename}")

if __name__ == '__main__':
    main()