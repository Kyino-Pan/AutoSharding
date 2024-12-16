# agent/agent.py
from pyexpat import features

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agent.network import Actor, Critic
from agent.replay_buffer import ReplayBuffer
from config import (
    SHARD_AMOUNT, GAMMA, TAU, BUFFER_CAPACITY, BATCH_SIZE, LEARNING_RATE,TOP_ACCOUNTS_PER_SHARD
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor Network
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.target_actor = Actor(state_dim, action_dim).to(DEVICE)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

        # Critic Network
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.target_critic = Critic(state_dim, action_dim).to(DEVICE)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

        # Exploration noise
        self.noise = OUNoise(action_dim*SHARD_AMOUNT*TOP_ACCOUNTS_PER_SHARD)

    def info_to_feature(self,state, accounts_info):

        m = _compute_m(state)
        # Compute n and get account_ids with fixed number
        n, account_ids = _compute_n(accounts_info)
        a = n.shape[0]

        if a == 0:
            # No accounts, no migration
            return np.zeros(self.action_dim, dtype=np.float32)

        # Compute features: a x (k*2)
        fs = []
        for i in range(a):
            account_vec = []
            for j in range(SHARD_AMOUNT):
                # 片内交易比例为正值，跨片交易比例为负值
                account_vec.append(n[i, j] * m[j, 0])  # 片内交易比例
                account_vec.append(n[i, j] * m[j, 1])  # 跨片交易比例（负值）
            fs.append(account_vec)
        fs = np.array(fs, dtype=np.float32)
        fs = torch.tensor(fs, dtype=torch.float32, device=DEVICE)  # (a, 16)
        return fs,account_ids

    def select_action(self, state, accounts_info, noise=True):
        # Compute m
        actor_features,accounts_ids = self.info_to_feature(state,accounts_info)
        a = actor_features.shape[0]
        # Pass each account's feature through Actor network to get action vectors
        self.actor.eval()
        with torch.no_grad():
            action_vectors = self.actor(actor_features)  # (a, 8)
        self.actor.train()

        # Add noise if required
        if noise:
            noise_sample = self.noise.sample()  # (action_dim,) e.g., (128x8=1024,)
            noise_sample = noise_sample.reshape(a, SHARD_AMOUNT)  # (a, 8)
            action_vectors += torch.tensor(noise_sample, dtype=torch.float32, device=DEVICE)

        # Clip actions to [0,1]
        action_vectors = torch.clamp(action_vectors, 0.0, 1.0)

        return action_vectors, accounts_ids

    def reset_noise(self):
        self.noise.reset()

    # agent/agent.py

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        # Sample a batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, accounts_info_batch = self.replay_buffer.sample(
            BATCH_SIZE)
        # Move to DEVICE
        state_batch = state_batch.to(DEVICE)
        action_batch = action_batch.to(DEVICE)
        reward_batch = reward_batch.to(DEVICE)
        next_state_batch = next_state_batch.to(DEVICE)
        done_batch = done_batch.to(DEVICE)

        # Compute target actions for the batch
        target_actions = []
        for i in range(BATCH_SIZE):
            next_state = next_state_batch[i].cpu().numpy()
            accounts_info = accounts_info_batch[i]
            critic_features, account_ids = self.info_to_feature(next_state,accounts_info)

            # Pass through target_actor to get action vectors
            with torch.no_grad():
                target_action_vectors = self.target_actor(critic_features)  # (a, 8)

            # Clip actions to [0,1]
            target_action_vectors = torch.clamp(target_action_vectors, 0.0, 1.0)

            # Convert to numpy
            target_action_vectors_np = target_action_vectors.cpu().numpy()  # (a, 8)

            # Assemble migration matrix
            migration_matrix = np.zeros((SHARD_AMOUNT, SHARD_AMOUNT), dtype=np.float32)
            for j, acc in enumerate(account_ids):
                from_shard = get_account_shard(acc, accounts_info)
                for k in range(SHARD_AMOUNT):
                    migration_matrix[from_shard, k] += target_action_vectors_np[j, k]

            # Flatten to get target action
            target_action = migration_matrix.flatten()  # (64,)

            target_actions.append(target_action)
        # Convert target_actions to tensor
        target_actions = torch.tensor(target_actions, dtype=torch.float32, device=DEVICE)  # (batch_size, 64)

        # Compute target Q-values
        with torch.no_grad():
            target_q = self.target_critic(next_state_batch, target_actions)
            target_q = reward_batch + (1 - done_batch) * GAMMA * target_q

        # Get current Q-values
        current_q = self.critic(state_batch, action_batch)

        # Compute Critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)

        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute Actor loss
        # 对于每个样本，从 state_batch 和 accounts_info_batch 中获取特征，不使用 torch.no_grad()
        actor_actions = []
        for i in range(BATCH_SIZE):
            # 从当前样本获取状态和账户信息
            state_np = state_batch[i].cpu().numpy()
            accounts_info = accounts_info_batch[i]

            # 使用 info_to_feature 方法获取 (critic_features, account_ids)
            # 假设 info_to_feature(state, accounts_info) 返回 (critic_features, account_ids)
            critic_features, account_ids = self.info_to_feature(state_np, accounts_info)
            # critic_features: (a,16) 张量, a表示账户数
            # account_ids: list of accounts

            # 不使用 torch.no_grad()，以便保留梯度
            action_vectors = self.actor(critic_features)  # (a, 8)
            action_vectors = torch.clamp(action_vectors, 0.0, 1.0)

            # 在 torch 中组装迁移矩阵
            migration_matrix = torch.zeros((SHARD_AMOUNT, SHARD_AMOUNT), dtype=torch.float32, device=DEVICE)

            # 将每个账户的动作累加到对应的from_shard行中
            # action_vectors[j,:] 为账户 j 对应的(8,)动作向量
            for j, acc in enumerate(account_ids):
                from_shard = get_account_shard(acc, accounts_info)
                # 将action_vectors[j,:]加到migration_matrix[from_shard,:]上
                migration_matrix[from_shard, :] += action_vectors[j, :]

            # 展平得到最终动作向量 (64,)
            action = migration_matrix.view(-1)  # (64,)
            actor_actions.append(action)

        # Convert actor_actions to tensor
        actor_actions = torch.stack(actor_actions, dim=0)  # (batch_size, 64)

        # Compute Actor loss
        # 此时 actor_actions 是通过 actor 网络计算并保持梯度
        actor_loss = -self.critic(state_batch, actor_actions).mean()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done, accounts_info):
        """
        存储转移，包括 accounts_info
        """
        # action is the flattened k*k migration vector
        self.replay_buffer.push(state, action, reward, next_state, done, accounts_info)


# Ornstein-Uhlenbeck process for exploration noise
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

def _compute_m(state):
    # state is k*k flattened matrix
    state_matrix = state.reshape((SHARD_AMOUNT, SHARD_AMOUNT))
    # Compute shard-internal and cross-shard ratios for each shard
    m = np.zeros((SHARD_AMOUNT, 2), dtype=np.float32)
    total_transactions = np.sum(state_matrix)
    for j in range(SHARD_AMOUNT):
        shard_total_tx = np.sum(state_matrix[j, :])
        intra_shard_tx = state_matrix[j, j]
        cross_shard_tx = shard_total_tx - intra_shard_tx
        intra_shard_ratio = intra_shard_tx / shard_total_tx if shard_total_tx > 0 else 0.0
        cross_shard_ratio = cross_shard_tx / shard_total_tx if shard_total_tx > 0 else 0.0
        m[j, 0] = intra_shard_ratio  # 片内交易比例为正值
        m[j, 1] = -cross_shard_ratio  # 跨片交易比例为负值
    return m  # shape k x 2


def _compute_n(accounts_info):
    # Fixed number of accounts per shard
    # Select top_accounts_per_shard by cross-shard ratio
    selected_accounts = {}
    for shard in range(SHARD_AMOUNT):
        # Filter accounts in this shard
        shard_accounts = {acc: info for acc, info in accounts_info.items() if info["shard"] == shard}
        if not shard_accounts:
            continue
        # Sort by cross-shard ratio
        sorted_by_ratio = sorted(shard_accounts.items(), key=lambda item:
                                 (item[1]["cross_shard_count"] / (item[1]["cross_shard_count"] + item[1]["intra_shard_count"] + 1e-6)),
                                 reverse=True)
        top_ratio_accounts = [acc for acc, info in sorted_by_ratio[:TOP_ACCOUNTS_PER_SHARD]]
        # Select up to TOP_ACCOUNTS_PER_SHARD
        selected_accounts[shard] = top_ratio_accounts

    # Now, build n matrix
    a = SHARD_AMOUNT * TOP_ACCOUNTS_PER_SHARD  # Fixed number of accounts
    n = np.zeros((a, SHARD_AMOUNT), dtype=np.float32)
    account_ids = []
    index = 0
    for shard in range(SHARD_AMOUNT):
        accounts = selected_accounts.get(shard, [])
        for acc in accounts:
            account_ids.append(acc)
            info = accounts_info[acc]
            total_tx = info["cross_shard_count"] + info["intra_shard_count"]
            if total_tx > 0:
                # Cross-shard distribution
                for target_shard, count in info["cross_shard_distribution"].items():
                    n[index, target_shard] += count
                # Intra-shard transactions
                home_shard = info["shard"]
                n[index, home_shard] += info["intra_shard_count"]
                # Normalize
                n[index, :] /= total_tx
            index += 1

    # If fewer accounts, the remaining rows stay zero (already initialized)
    return n, account_ids  # n: a x k, account_ids: list of a accounts

def get_account_shard(account_id, accounts_info):
    # Helper method to get the shard of a given account
    if account_id in accounts_info:
        return accounts_info[account_id]["shard"]
    else:
        # 默认分配到分片0或其他处理方式
        panic()
        return 0
