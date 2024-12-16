# agent/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_dim=16, output_dim=8):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入特征，形状为 (batch_size, 16)
        :return: 动作向量，形状为 (batch_size, 8)
        """
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = torch.sigmoid(self.fc3(x))  # 输出在 [0,1] 范围内
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 假设 state_dim=64（8x8 分片状态向量），action_dim=64（8x8 迁移矩阵）
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        """
        前向传播函数
        :param state: 状态向量，形状为 (batch_size, state_dim)
        :param action: 动作向量，形状为 (batch_size, action_dim)
        :return: Q值，形状为 (batch_size, 1)
        """
        xa = torch.cat([state, action], dim=1)  # 拼接状态和动作，形状为 (batch_size, state_dim + action_dim)
        xa = F.leaky_relu(self.fc1(xa), negative_slope=0.2)
        xa = F.leaky_relu(self.fc2(xa), negative_slope=0.2)
        q = self.fc3(xa)
        return q
