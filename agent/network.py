# agent/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BATCH_SIZE, TOP_ACCOUNTS_PER_SHARD


class Actor(nn.Module):
    def __init__(self, input_dim=16, output_dim=8):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = torch.softmax(self.fc3(x),dim=1)  # 输出在 [0,1] 范围内
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, acc_amount=128):
        super(Critic, self).__init__()
        self.acc_amount = acc_amount  # 每个样本的子维度数量

        # 定义网络层（共享权重）
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        """
        前向传播函数
        :param state: 状态张量，形状为 (batch_size, acc_amount, state_dim)
        :param action: 动作张量，形状为 (batch_size, acc_amount, action_dim)
        :return: Q值，形状为 (batch_size, 1)
        """
        # 合并状态和动作的最后一维
        xa = torch.cat([state, action], dim=2)  # 形状为 (batch_size, acc_amount, state_dim + action_dim)

        # 批量维度内逐元素处理，使用共享的全连接网络
        batch_size, acc_amount, _ = xa.shape
        xa = xa.view(batch_size * acc_amount, -1)  # 展平为 (batch_size * acc_amount, state_dim + action_dim)

        # 网络前向传播
        xa = F.leaky_relu(self.fc1(xa), negative_slope=0.2)
        xa = F.leaky_relu(self.fc2(xa), negative_slope=0.2)
        xa = self.fc3(xa)  # 输出形状为 (batch_size * acc_amount, 1)

        # 恢复原始批量维度
        xa = xa.view(batch_size, acc_amount, -1)  # 恢复为 (batch_size, acc_amount, 1)

        # 聚合 acc_amount 维度，计算平均值（或总和）
        q = torch.mean(xa, dim=1)  # 输出形状为 (batch_size, 1)

        return q
