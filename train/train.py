# train.py

import torch
import numpy as np
from env.environment import EthEnvironment
from agent.agent import DDPGAgent
from config import (
    SHARD_AMOUNT, STATE_WINDOW, DATA_FILE, SEED, ACTIVATE_THRESHOLD,
    MIGRATION_COST, INCENTIVE, PENALTY_COEFF, REWARD_DELAY,
    GAMMA, TAU, BUFFER_CAPACITY, BATCH_SIZE, LEARNING_RATE,
    TOP_ACCOUNTS_PER_SHARD  # 确保引入
)

def main():
    env = EthEnvironment()
    state_dim = SHARD_AMOUNT * 2
    action_dim = SHARD_AMOUNT   # Migration matrix flattened
    agent = DDPGAgent(state_dim, action_dim)

    num_episodes = 100  # 训练的总轮数
    max_steps = 10000    # 每轮的最大步数

    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_noise()  # 重置噪声
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            # Agent选择动作
            action_vector, account_ids = agent.select_action(state, env.accounts_info, noise=True)  # (k*k,), list of a accounts

            # 将action_vector转换为迁移动作列表
            # 设定一个阈值决定是否迁移
            migration_threshold = 0.5
            migrations = []
            migration_matrix = action_vector.reshape((SHARD_AMOUNT, SHARD_AMOUNT))
            for from_shard in range(SHARD_AMOUNT):
                for to_shard in range(SHARD_AMOUNT):
                    if from_shard != to_shard and migration_matrix[from_shard, to_shard] > migration_threshold:
                        # 找出属于from_shard的账户索引
                        for acc_index, acc in enumerate(account_ids):
                            acc_shard = env.accounts_info.get(acc, {}).get("shard", from_shard)
                            if acc_shard == from_shard:
                                migrations.append((acc, to_shard))
            # 执行迁移
            next_state, reward, done, info = env.step(migrations)

            # 存储经验
            agent.store_transition(state, action_vector, reward, next_state, done, env.accounts_info)

            # 更新Agent
            agent.update()

            state = next_state
            total_reward += reward
            step += 1

        print(f"Episode {episode +1 }, Total Reward: {total_reward}, Steps: {step}")

    env.close()


if __name__ == "__main__":
    main()
