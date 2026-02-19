import torch
import torch.nn as nn
import numpy as np
from minatar import Environment
import time
import matplotlib.pyplot as plt

# 1. Architecture: Must match the training script exactly
class PolicyNetwork(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(PolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[2], 16, kernel_size=3, padding=1)
        )
        self.actor = nn.Sequential(
            nn.Linear(1600, 64), 
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(1600, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1) 
        return self.actor(x), self.critic(x)

def run_95_score_demo():
    # Initialize MinAtar Space Invaders
    env = Environment('space_invaders')
    state_shape = env.state_shape()
    n_actions = env.num_actions()
    
    model = PolicyNetwork(state_shape, n_actions)
    
    # 2. Load the peak model trained at 10,000 iterations
    try:
        # Loading from the root project directory
        checkpoint = torch.load("best_ppo_model_peak.pth")
        model.load_state_dict(checkpoint)
        model.eval()
        print("--- [SUCCESS] 95-Point Master Model Loaded Successfully ---")
    except Exception as e:
        print(f"--- [ERROR] Model loading failed: {e} ---")
        return

    # Visualization settings
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Space Invaders: 95 Peak Score Performance")
    img_handle = None

    env.reset()
    state = env.state()
    total_reward = 0
    done = False
    
    print("--- READY: Press Win + Alt + R to start recording NOW! ---")
    
    while not done:
        # Data preparation (H, W, C) -> (Batch, C, H, W)
        s_np = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(s_np).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            probs, _ = model(state_tensor)
        
        # Select best action (Argmax) for optimal demonstration
        action = torch.argmax(probs).item()
        
        # Execute action in environment
        reward, done = env.act(action)
        state = env.state()
        total_reward += reward
        
        # Rendering logic
        frame = np.max(state, axis=2) 
        if img_handle is None:
            img_handle = ax.imshow(frame, cmap='magma')
        else:
            img_handle.set_data(frame)
        
        # Pause to allow for clear video recording (adjust as needed)
        plt.pause(0.1) 
        
    print(f"--- Demonstration Finished | Final Mastery Score: {total_reward} ---")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_95_score_demo()