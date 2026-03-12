import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import collections
import os
from minatar import Environment

# ==========================================================
# EXPERIMENT #8004: FINAL RESEARCH-GRADE PPO (ALIGNED)
# Metrics: V1 Environment + Minimal Actions + FrameStack 4
# Stopping: Exactly 1,000,000 Environment Steps
# Improvements: GAE Index-Fix + Advantage Norm + Path Auto-Create
# ==========================================================
LR_START = 0.0003
LR_END = 0.00005
ENTROPY_START = 0.01
ENTROPY_END = 0.001

K_EPOCHS = 4
BATCH_SIZE = 64
GAMMA = 0.99
LMBDA = 0.95
EPS_CLIP = 0.2
MAX_STEPS = 1000000 
STACK_SIZE = 4
CSV_FILE = '../data/ppo_progress_8004.csv' 
# ==========================================================

class MinAtarFrameStack:
    def __init__(self, env, k=4):
        self.env = env
        self.k = k
        self.frames = collections.deque([], maxlen=k)
        shp = env.state_shape()
        self.observation_space_shape = (shp[2] * k, shp[0], shp[1])

    def reset(self):
        self.env.reset()
        state = self.env.state() 
        for _ in range(self.k):
            self.frames.append(state)
        return self._get_stack()

    def step(self, action):
        reward, done = self.env.act(action)
        state = self.env.state()
        self.frames.append(state)
        return self._get_stack(), reward, done, None

    def _get_stack(self):
        return np.concatenate(list(self.frames), axis=2).transpose(2, 0, 1)

    def num_actions(self):
        return self.env.num_actions()

    def state_shape(self):
        return self.observation_space_shape

class ActorCritic(nn.Module):
    def __init__(self, num_actions, in_channels):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.policy = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.conv(x)
        return self.policy(features), self.value(features)

class Agent:
    def __init__(self, num_actions, in_channels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(num_actions, in_channels).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_START)
        self.buffer = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, val = self.model(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), val.item()

    def store(self, state, action, prob, val, reward, done):
        self.buffer.append((state, action, prob, val, reward, done))

    def learn(self, ent_coef):
        if len(self.buffer) == 0: return
        states, actions, old_probs, values, rewards, masks = zip(*self.buffer)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # --- GAE CALCULATION (TEAMMATE'S INDEX FIX) ---
        returns = []
        gae = 0
        next_value = 0 
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_value * (1 - masks[t]) - values[t]
            gae = delta + GAMMA * LMBDA * (1 - masks[t]) * gae
            returns.insert(0, gae + values[t])
            next_value = values[t]

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - values
        # Advantage Normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # ----------------------------------------------

        for _ in range(K_EPOCHS):
            probs, val_preds = self.model(states)
            dist = Categorical(probs)
            new_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_probs - old_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - val_preds.squeeze()).pow(2).mean() - ent_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        self.buffer = []

if __name__ == '__main__':
    # --- AUTO-CREATE DIRECTORY ---
    data_dir = os.path.dirname(CSV_FILE)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    raw_env = Environment('space_invaders', sticky_action_prob=0.0)
    MINIMAL_ACTIONS = raw_env.minimal_action_set()
    NUM_ACTS = len(MINIMAL_ACTIONS) 
    env = MinAtarFrameStack(raw_env, k=STACK_SIZE)
    agent = Agent(NUM_ACTS, env.state_shape()[0])

    data_list = []
    total_env_steps = 0  
    iteration = 0
    
    print(f"--- [START] Final Aligned PPO Run ---")
    while total_env_steps < MAX_STEPS:
        frac = total_env_steps / MAX_STEPS
        current_lr = max(LR_START - frac * (LR_START - LR_END), LR_END)
        current_ent = max(ENTROPY_START - frac * (ENTROPY_START - ENTROPY_END), ENTROPY_END)
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = current_lr
            
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action_idx, prob, val = agent.choose_action(obs)
            real_action = MINIMAL_ACTIONS[action_idx]
            next_obs, reward, done, _ = env.step(real_action)
            agent.store(obs, action_idx, prob, val, reward, done)
            obs = next_obs
            episode_reward += reward
            total_env_steps += 1 
        
        agent.learn(current_ent)
        data_list.append([total_env_steps, episode_reward])
        if iteration % 100 == 0:
            print(f"Steps: {total_env_steps/1000:.1f}K/1000K | Reward: {episode_reward}")
            pd.DataFrame(data_list, columns=['Total_Steps', 'Reward']).to_csv(CSV_FILE, index=False)
        iteration += 1
    print(f"--- [COMPLETE] Target Reached ---")
