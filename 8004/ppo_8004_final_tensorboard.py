import os
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from minatar import Environment

# ==========================================================
# EXPERIMENT #8004: FINAL RESEARCH-GRADE PPO (FIXED)
# TensorBoard-enabled version
# Reward is explicitly logged against TOTAL ENVIRONMENT STEPS.
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
TB_PARENT_DIR = 'PPO'
RUN_NAME = f"ppo_8004_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TB_LOG_DIR = os.path.join(TB_PARENT_DIR, RUN_NAME)
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        if len(self.buffer) == 0:
            return None

        states, actions, old_probs, values, rewards, masks = zip(*self.buffer)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        returns = []
        gae = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_value * (1 - masks[t]) - values[t]
            gae = delta + GAMMA * LMBDA * (1 - masks[t]) * gae
            returns.insert(0, gae + values[t])
            next_value = values[t]

        returns = torch.FloatTensor(returns).to(self.device)
        raw_advantages = returns - values
        advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)

        metrics = {}
        for _ in range(K_EPOCHS):
            probs, val_preds = self.model(states)
            dist = Categorical(probs)
            new_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_probs - old_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - val_preds.squeeze()).pow(2).mean()
            loss = policy_loss + value_loss - ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            metrics = {
                'loss/total': loss.item(),
                'loss/policy': policy_loss.item(),
                'loss/value': value_loss.item(),
                'policy/entropy': entropy.item(),
                'policy/ratio_mean': ratio.mean().item(),
                'advantages/raw_mean': raw_advantages.mean().item(),
                'advantages/raw_std': raw_advantages.std().item(),
                'returns/mean': returns.mean().item(),
            }

        self.buffer = []
        return metrics


if __name__ == '__main__':
    data_dir = os.path.dirname(CSV_FILE)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir)

    os.makedirs(TB_LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TB_LOG_DIR)

    raw_env = Environment('space_invaders', sticky_action_prob=0.0)
    MINIMAL_ACTIONS = raw_env.minimal_action_set()
    NUM_ACTS = len(MINIMAL_ACTIONS)
    env = MinAtarFrameStack(raw_env, k=STACK_SIZE)
    agent = Agent(NUM_ACTS, env.state_shape()[0])

    data_list = []
    total_env_steps = 0
    iteration = 0

    print('--- [START] Final Aligned PPO Run (1000K Target) ---')
    print(f'TensorBoard log directory: {TB_LOG_DIR}')

    try:
        while total_env_steps < MAX_STEPS:
            frac = total_env_steps / MAX_STEPS
            current_lr = max(LR_START - frac * (LR_START - LR_END), LR_END)
            current_ent = max(ENTROPY_START - frac * (ENTROPY_START - ENTROPY_END), ENTROPY_END)
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = current_lr

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done:
                action_idx, prob, val = agent.choose_action(obs)
                real_action = MINIMAL_ACTIONS[action_idx]
                next_obs, reward, done, _ = env.step(real_action)
                agent.store(obs, action_idx, prob, val, reward, done)
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_env_steps += 1

            learn_metrics = agent.learn(current_ent)
            data_list.append([total_env_steps, episode_reward])

            # IMPORTANT: reward uses total_env_steps as the x-axis in TensorBoard.
            writer.add_scalar('charts/episode_reward_vs_total_steps', episode_reward, total_env_steps)
            writer.add_scalar('charts/episode_length_vs_total_steps', episode_steps, total_env_steps)
            writer.add_scalar('charts/learning_rate_vs_total_steps', current_lr, total_env_steps)
            writer.add_scalar('charts/entropy_coef_vs_total_steps', current_ent, total_env_steps)
            writer.add_scalar('charts/episodes_vs_total_steps', iteration + 1, total_env_steps)

            if learn_metrics is not None:
                for tag, value in learn_metrics.items():
                    writer.add_scalar(tag, value, total_env_steps)

            if iteration % 100 == 0:
                print(f'Steps: {total_env_steps/1000:.1f}K/1000K | Reward: {episode_reward}')
                pd.DataFrame(data_list, columns=['Total_Steps', 'Reward']).to_csv(CSV_FILE, index=False)
                writer.flush()

            iteration += 1

        pd.DataFrame(data_list, columns=['Total_Steps', 'Reward']).to_csv(CSV_FILE, index=False)
        writer.flush()
        print(f'--- [COMPLETE] Target Reached: {total_env_steps} steps. ---')
    finally:
        writer.close()
