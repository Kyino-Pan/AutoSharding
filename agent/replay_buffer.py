# agent/replay_buffer.py

import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        存储转移，包括 accounts_info
        """
        # 转换为numpy数组并存储
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        reward = float(reward)
        done = float(done)
        # accounts_info 应该是一个字典或可以序列化的结构
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        # 转换为torch张量
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        # accounts_info 是一个列表，每个元素是一个字典
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
