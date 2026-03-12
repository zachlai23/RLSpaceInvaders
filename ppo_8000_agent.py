import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import os
import collections
from minatar import Environment

# ==========================================
#      EXPERIMENT #8000: PERFORMANCE OPTIM
# ==========================================
# Decay Targets
LR_START = 0.0003      # Original LR
LR_END = 0.00005       # Stable finish
ENTROPY_START = 0.01   # Initial exploration
ENTROPY_END = 0.001    # Late-game precision

K_EPOCHS = 4
BATCH_SIZE = 64
GAMMA = 0.99
LMBDA = 0.95
EPS_CLIP = 0.2         #
MAX_ITERATIONS = 10001 # Extended training
STACK_SIZE = 4
CSV_FILE = 'ppo_progress_plot_8000.csv'
# ==========================================

class MinAtarFrameStack:
    def __init__(self, env, k=4):
        self.env = env
        self.k = k
        self.frames = collections.deque([], maxlen=k)
        shp = env.state_shape()
        self.observation_space_shape = (shp[2] * k, shp[0], shp[1])

    def reset(self):
        self.env.reset()
        obs = self.env.state()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        reward, done = self.env.act(action)
        obs = self.env.state()
        self.frames.append(obs)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        stacked = np.concatenate(list(self.frames), axis=2)
        return np.transpose(stacked, (2, 0, 1))

    def state_shape(self):
        return self.observation_space_shape
    
    def __getattr__(self, name):
        return getattr(self.env, name)

class ActorCriticCNN(nn.Module):
    def __init__(self, n_actions, in_channels):
        super(ActorCriticCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def evaluate(self, state, action):
        features = self.fc(self.encoder(state))
        probs = torch.softmax(self.actor(features), dim=-1)
        dist = Categorical(probs)
        return dist.log_prob(action), self.critic(features), dist.entropy()

    def act(self, state):
        features = self.fc(self.encoder(state))
        probs = torch.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return probs, value

class Agent:
    def __init__(self, n_actions, in_channels):
        self.actor_critic = ActorCriticCNN(n_actions, in_channels)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LR_START)
        self.memory = {'states':[], 'actions':[], 'probs':[], 'vals':[], 'rewards':[], 'dones':[]}

    def choose_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, value = self.actor_critic.act(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def store(self, state, action, prob, val, reward, done):
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['probs'].append(prob)
        self.memory['vals'].append(val)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

    def learn(self, entropy_coeff):
        states = torch.FloatTensor(np.array(self.memory['states']))
        actions = torch.LongTensor(np.array(self.memory['actions']))
        old_probs = torch.FloatTensor(np.array(self.memory['probs']))
        old_vals = torch.FloatTensor(np.array(self.memory['vals']))
        
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory['rewards']), reversed(self.memory['dones'])):
            if done: discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns)
        advantages = returns - old_vals

        for _ in range(K_EPOCHS):
            log_probs, state_values, dist_entropy = self.actor_critic.evaluate(states, actions)
            ratios = torch.exp(log_probs - old_probs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            
            # Use dynamic entropy_coeff for loss calculation
            loss = -torch.min(surr1, surr2) + 0.5 * (state_values.squeeze() - returns)**2 - entropy_coeff * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        for key in self.memory: self.memory[key] = []

if __name__ == '__main__':
    env = MinAtarFrameStack(Environment('space_invaders'), k=STACK_SIZE)
    agent = Agent(env.num_actions(), env.state_shape()[0])
    data_list = []
    
    print(f"--- Starting Experiment #8000: Dual-Decay Optimization ---")
    
    for i in range(MAX_ITERATIONS):
        # 1. Update Linear Decay Ratios
        frac = i / MAX_ITERATIONS
        current_lr = LR_START - frac * (LR_START - LR_END)
        current_ent = ENTROPY_START - frac * (ENTROPY_START - ENTROPY_END)
        
        # 2. Update Optimizer Learning Rate
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = current_lr
            
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, prob, val = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.store(obs, action, prob, val, reward, done)
            obs = next_obs
            total_reward += reward
        
        # 3. Learn with current entropy coefficient
        agent.learn(current_ent)
        data_list.append([i, total_reward])
        
        if i % 100 == 0:
            print(f"Iteration {i} | Reward: {total_reward} | LR: {current_lr:.6f} | Ent: {current_ent:.4f}")
            pd.DataFrame(data_list, columns=['iteration', 'reward']).to_csv(CSV_FILE, index=False)

    print("Experiment #8000 Complete. Data saved to", CSV_FILE)