# train.py

import numpy as np

from agent.agent import DDPGAgent, get_account_shard
from config import (
    SHARD_AMOUNT,
    REWARD_DELAY,
    MIGRATION_MIN_AMOUNT, MIGRATION_MAX_AMOUNT  # 确保引入
)
from env.environment import EthEnvironment


def main():
    env = EthEnvironment()
    state_dim = SHARD_AMOUNT * 2
    action_dim = SHARD_AMOUNT  # Migration matrix flattened
    agent = DDPGAgent(state_dim, action_dim)

    num_episodes = 100  # 训练的总轮数
    max_steps = 200000  # 每轮的最大步数

    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_noise()  # 重置噪声
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            # Agent选择动作
            actor_features, accounts_ids = agent.info_to_feature(state, env.accounts_info)
            action_vectors = agent.select_action(actor_features, noise=True)  # (k*k,), list of a accounts
            migration_threshold = 0.65
            migrations = []
            mig_thresh = ((float(max_steps-step)/max_steps)*MIGRATION_MIN_AMOUNT
                          + (float(step)/max_steps) * MIGRATION_MAX_AMOUNT)
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
            if len(migrations) > mig_thresh:
                # Execute migration
                next_state, reward, done = env.step(migrations)
                next_feature,_ = agent.info_to_feature(next_state, env.accounts_info)
                step += REWARD_DELAY
                # 存储经验
                agent.store_transition(actor_features, action_vectors, reward, next_feature, done)
            else:
                next_state, reward, done = env.step([])
                step += 1
            # 更新Agent
            agent.update()
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Steps: {step}")

    env.close()


if __name__ == "__main__":
    main()
