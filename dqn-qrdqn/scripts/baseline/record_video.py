import gymnasium as gym
import numpy as np
import imageio
from gymnasium import spaces
from stable_baselines3 import DQN
from minatar import Environment
from pathlib import Path

# Environment Wrapper
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

# Convert 10x10 grid to video frames, final video 300x300
def get_colored_frame(state, scale=30):
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    colors = [
        [0, 255, 0],    # Channel 0: Cannon (Player) -> Green
        [255, 0, 0],    # Channel 1: Aliens          -> Red
        [100, 0, 0],    # Channel 2: Alien Trails    -> Dark Red
        [0, 0, 255],    # Channel 3: Shields         -> Blue
        [255, 255, 255],# Channel 4: Player Bullets  -> White
        [255, 255, 0]   # Channel 5: Alien Bullets   -> Yellow
    ]

    for channel in range(6): 
        if channel < state.shape[2]:
            mask = state[:, :, channel]
            for c in range(3): # R, G, B
                img[:, :, c] = np.maximum(img[:, :, c], mask * colors[channel][c])

    big_img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return big_img

# Main recording loop
def main():
    MODEL_NAME = "DQN_Baseline_1mil"

    # Load and save paths
    root_dir = Path(__file__).resolve().parents[2]
    model_load_path = root_dir / "models" / "baseline" / MODEL_NAME
    video_save_dir = root_dir / "videos" / "baseline"
    
    video_save_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = MinAtarLocalEnv("space_invaders")
    
    # Load trained model
    print(f"Loading model '{MODEL_NAME}' from {model_load_path}...")
    try:
        model = DQN.load(str(model_load_path))
    except Exception as e:
        print(f"Error: Could not find '{model_load_path}.zip'. Did you run training?")
        return

    N_VIDEOS = 1  

    for i in range(1, N_VIDEOS + 1):
        obs, _ = env.reset()
        done = False
        frames = []
        score = 0
        
        print(f"Recording Game {i}...")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            
            current_grid = env.game.state()
            frames.append(get_colored_frame(current_grid))

        # Save video
        filename = f"{MODEL_NAME}_game_{i}_score_{int(score)}.mp4"
        save_path = video_save_dir / filename
        
        print(f"  -> Game Over! Score: {score}. Saving {len(frames)} frames to {save_path}...")
        imageio.mimsave(str(save_path), frames, fps=30)

    print("\nVideo saved successfully!")

if __name__ == '__main__':
    main()