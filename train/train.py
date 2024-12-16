# train.py

import torch
import numpy as np
from env.environment import EthEnvironment
from agent.agent import DDPGAgent, get_account_shard
from config import (
    SHARD_AMOUNT, STATE_WINDOW, DATA_FILE, SEED, ACTIVATE_THRESHOLD,
    MIGRATION_COST, INCENTIVE, PENALTY_COEFF, REWARD_DELAY,
    GAMMA, TAU, BUFFER_CAPACITY, BATCH_SIZE, LEARNING_RATE,
    TOP_ACCOUNTS_PER_SHARD, MIGRATION_THRESHOLD  # 确保引入
)


def main():
    env = EthEnvironment()
    state_dim = SHARD_AMOUNT * 2
    action_dim = SHARD_AMOUNT  # Migration matrix flattened
    agent = DDPGAgent(state_dim, action_dim)

    num_episodes = 100  # 训练的总轮数
    max_steps = 10000  # 每轮的最大步数

    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_noise()  # 重置噪声
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            # Agent选择动作
            action_vectors, accounts_ids = agent.select_action(state, env.accounts_info,
                                                               noise=True)  # (k*k,), list of a accounts
            migration_threshold = 0.854
            migrations = []

            # Iterate over each account and its corresponding action vector
            for acc_index, acc in enumerate(accounts_ids):
                action_vector = action_vectors[acc_index].cpu().numpy()  # (8,)
                from_shard = get_account_shard(acc, env.accounts_info)

                # Determine target shard via argmax
                to_shard = np.argmax(action_vector)

                # If target shard is the same as current shard, skip
                if to_shard == from_shard:
                    continue
                # Check if the action value for the target shard exceeds the threshold
                if action_vector[to_shard] > migration_threshold:
                    migrations.append((acc, to_shard))
            if len(migrations) > MIGRATION_THRESHOLD:
                # Execute migration
                next_state, reward, done = env.step(migrations)
            else:
                next_state, reward, done = env.step([])
            # 存储经验
            agent.store_transition(state, action_vectors, reward, next_state, done, env.accounts_info)
            # 更新Agent
            agent.update()

            state = next_state
            total_reward += reward
            step += 1

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Steps: {step}")

    env.close()


if __name__ == "__main__":
    main()
