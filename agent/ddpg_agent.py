import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import SHARD_AMOUNT
from agent.network import MLP


class DDPGActor(nn.Module):
    def __init__(self, input_dim, k=SHARD_AMOUNT):
        super(DDPGActor, self).__init__()
        # 输入维度例如为 (k*k) + ... 或者使用env构建的状态
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, k)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出为k维，使用softmax获得分布
        # DDPG中通常不使用softmax，但这里为方便
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class DDPGAgent:
    def __init__(self, k=SHARD_AMOUNT, lr=1e-3):
        self.k = k
        self.actor = DDPGActor(input_dim=k * k, k=k)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        # DDPG还需Critic网络、target网络、replay buffer等，这里略

    def act(self, state, accounts_info):
        # 与之前的agent逻辑类似，先构建m,n，得到特征
        # 简化：state为k*k维，accounts_info为dictionary
        # 假设已经有特征提取逻辑在此实现(和之前的Agent类似)
        # 以下为伪代码:
        # 1. 将state reshape成k×k
        # 2. 根据m,n逻辑提取a个账户的特征(2*k维)
        # 对于DDPG Actor，我们简单处理，只对ENV的state进行决策（简化）

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.actor(state_t)  # (1, k)
        probs = probs.detach().numpy()[0]
        # 对所有账户统一选择同一分片（非常简化，不合理）
        # 实际应像前面那样对每账户构建特征，再分别通过actor决策
        # 这里仅演示DDPG actor使用方式，实际需要和之前逻辑整合
        chosen_shard = np.argmax(probs)

        # 对所有账户进行动作决策
        actions = []
        for acc, info in accounts_info.items():
            if info["shard"] != chosen_shard:
                actions.append((acc, chosen_shard))
        return actions

    def train(self, batch):
        # DDPG训练逻辑：更新Actor和Critic
        # 此处省略具体实现
        pass
