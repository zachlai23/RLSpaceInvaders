import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import collections
from minatar import Environment

# ==========================================
#   EXP #9000: THE APEX PROTOCOL
#   Objective: Maximum Peak Exploration
#   DNA: Pure #8000 + Ortho-Init + Frame Skip
# ==========================================
MAX_ITERATIONS = 20001    # User Override: 20K to allow massive exploration window
FRAME_SKIP = 4            # Temporal consistency for high-speed evasion

# Pure #8000 Dual Decay DNA (Stretched over 20K steps for prolonged exploration)
LR_START = 3e-4           
LR_END = 5e-5             
ENT_START = 0.01          
ENT_END = 0.001           

EPS_CLIP = 0.2
K_EPOCHS = 3
STACK_SIZE = 4
CSV_FILE = 'ppo_progress_9000_apex.csv'
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
        for _ in range(self.k): self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        total_reward = 0
        done = False
        # [APPLIED TECH] Frame Skipping: 1 Action = 4 Frames of execution
        for _ in range(FRAME_SKIP):
            if not done:
                r, d = self.env.act(action)
                total_reward += r  # PURE RAW SCORE. No survival bonus. No cowardice.
                done = d
        obs = self.env.state()
        self.frames.append(obs)
        return self._get_obs(), total_reward, done, {}

    def _get_obs(self):
        stacked = np.concatenate(list(self.frames), axis=2)
        return np.transpose(stacked, (2, 0, 1))

    def state_shape(self): return self.observation_space_shape
    def __getattr__(self, name): return getattr(self.env, name)

# [APPLIED TECH] Orthogonal Initialization: Keeps early gradients healthy
def layer_init(layer, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer

class ActorCriticCNN(nn.Module):
    def __init__(self, n_actions, in_channels):
        super(ActorCriticCNN, self).__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 3, 1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 2, 1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 1, 1)), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(layer_init(nn.Linear(3136, 512)), nn.ReLU())
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def act(self, state):
        features = self.fc(self.encoder(state))
        return torch.softmax(self.actor(features), -1), self.critic(features)

    def evaluate(self, state, action):
        features = self.fc(self.encoder(state))
        dist = Categorical(torch.softmax(self.actor(features), -1))
        return dist.log_prob(action), self.critic(features), dist.entropy()

class Agent:
    def __init__(self, n_actions, in_channels):
        self.ac = ActorCriticCNN(n_actions, in_channels)
        self.opt = optim.Adam(self.ac.parameters(), lr=LR_START)
        self.memory = {'s':[], 'a':[], 'p':[], 'r':[], 'd':[]}

    def choose_action(self, state):
        with torch.no_grad():
            probs, _ = self.ac.act(torch.FloatTensor(state).unsqueeze(0))
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def learn(self, ent_c):
        states = torch.FloatTensor(np.array(self.memory['s']))
        actions = torch.LongTensor(np.array(self.memory['a']))
        old_probs = torch.FloatTensor(np.array(self.memory['p']))
        
        returns, dr = [], 0
        for r, d in zip(reversed(self.memory['r']), reversed(self.memory['d'])):
            if d: dr = 0
            dr = r + (0.99 * dr)
            returns.insert(0, dr)
        returns = torch.FloatTensor(returns)

        # Pure aggressive PPO update (No strict advantage clipping that forces stagnation)
        for _ in range(K_EPOCHS):
            log_p, val, ent = self.ac.evaluate(states, actions)
            adv = (returns - val.squeeze().detach())
            ratio = torch.exp(log_p - old_probs)
            s1, s2 = ratio * adv, torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * adv
            loss = -torch.min(s1, s2) + 0.5 * (val.squeeze() - returns)**2 - ent_c * ent
            self.opt.zero_grad(); loss.mean().backward(); self.opt.step()
        for k in self.memory: self.memory[k] = []

if __name__ == '__main__':
    env = MinAtarFrameStack(Environment('space_invaders'), k=STACK_SIZE)
    agent = Agent(env.num_actions(), env.state_shape()[0])
    history = []
    
    print("--- INITIATING EXP #9000: 20K APEX MAX-EXPLORATION ---")
    for i in range(MAX_ITERATIONS):
        frac = i / MAX_ITERATIONS
        # The 20K stretch: Entropy and LR decay much slower, allowing huge early exploration
        curr_lr = LR_START - frac * (LR_START - LR_END)
        curr_ent = ENT_START - frac * (ENT_START - ENT_END)
        for g in agent.opt.param_groups: g['lr'] = curr_lr

        s, d, score = env.reset(), False, 0
        while not d:
            a, p = agent.choose_action(s)
            s_n, r, d, _ = env.step(a)
            agent.memory['s'].append(s); agent.memory['a'].append(a); agent.memory['p'].append(p)
            agent.memory['r'].append(r); agent.memory['d'].append(d)
            s, score = s_n, score + r
        
        agent.learn(curr_ent)
        history.append([i, score])
        
        if i % 100 == 0:
            print(f"Iter {i:5d} | Raw Peak Score: {score:4.1f} | LR: {curr_lr:.6f} | Ent: {curr_ent:.4f}")
            pd.DataFrame(history, columns=['iteration','reward']).to_csv(CSV_FILE, index=False)