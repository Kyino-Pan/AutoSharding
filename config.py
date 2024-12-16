# project/config.py
import torch

SHARD_AMOUNT = 8  # 分片数量k
STATE_WINDOW = 720  # 生成策略最少需要的的块数（7200 block/day）
SEED = 42

ACTIVATE_THRESHOLD = 1024  # 超过此块数未活跃则移除账户
REWARD_DELAY = 1024  # 动作执行后延迟50个块计算奖励

# 可以对训练过程超参数配置
NUM_EPISODES = 100
STEPS_PER_EPISODE = 100
BATCH_SIZE = 64
LR = 1e-3

# 假设我们有固定的设备、优化器等外部设定
DEVICE = torch.device("cpu")

MIGRATION_COST = 1.0      # 每次迁移动作的成本
INCENTIVE = 0.5           # 激励项， <= MIGRATION_COST
PENALTY_COEFF = 10.0      # 惩罚系数，用于频繁提案惩罚(惩罚 = PENALTY_COEFF * (1/interval))

# 其他参数如文件路径
DATA_FILE = "./data/eth_2.csv"

# DDPG相关配置
GAMMA = 0.99              # 折扣因子
TAU = 0.005               # 软更新参数
BUFFER_CAPACITY = 100000  # Replay Buffer容量
BATCH_SIZE = 64           # 训练批次大小
LEARNING_RATE = 1e-3      # 学习率

TOP_ACCOUNTS_PER_SHARD = 16
print(f"{SHARD_AMOUNT} shards are set, {TOP_ACCOUNTS_PER_SHARD} accounts in each shard are monitored. \n")

