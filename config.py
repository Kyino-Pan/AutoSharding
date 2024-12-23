# project/config.py
import torch

SHARD_AMOUNT = 8  # 分片数量k
STATE_WINDOW = 3000  # 生成策略最少需要的的块数
SEED = 42
BLOCK_PER_DAY = 7200

ACTIVATE_THRESHOLD = 4096  # 超过此块数未活跃则移除账户
REWARD_DELAY = 4096  # 动作执行后延迟4096
# 个块计算奖励

# 可以对训练过程超参数配置
NUM_EPISODES = 100
STEPS_PER_EPISODE = 100
LR = 1e-3

# 假设我们有固定的设备、优化器等外部设定
DEVICE = torch.device("cpu")

MIGRATION_COST = 0.00001      # 每次迁移动作的成本
PENALTY_COEFF = 50.0      # 惩罚系数，用于频繁提案惩罚(惩罚 = PENALTY_COEFF * (1/interval))

# 其他参数如文件路径
DATA_FILE = "./data/eth_2.csv"

# DDPG相关配置
GAMMA = 0.99              # 折扣因子
TAU = 0.005               # 软更新参数
BUFFER_CAPACITY = 100000  # Replay Buffer容量
BATCH_SIZE = 8           # 训练批次大小
LEARNING_RATE = 1e-3      # 学习率

TOP_ACCOUNTS_PER_SHARD = 32
MIGRATION_MIN_AMOUNT = SHARD_AMOUNT*TOP_ACCOUNTS_PER_SHARD/2
MIGRATION_MAX_AMOUNT = MIGRATION_MIN_AMOUNT
print(f"{SHARD_AMOUNT} shards are set, {TOP_ACCOUNTS_PER_SHARD} accounts in each shard are monitored. \n")

