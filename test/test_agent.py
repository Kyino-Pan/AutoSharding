# test/test_agent.py

import unittest
import os
import numpy as np

from config import SHARD_AMOUNT, DATA_FILE, BATCH_SIZE
from env.environment import EthEnvironment
from agent.agent import DDPGAgent

class TestDDPGAgent(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.exists(DATA_FILE), f"Data file {DATA_FILE} not found.")
        self.env = EthEnvironment()
        state_dim = SHARD_AMOUNT * SHARD_AMOUNT
        action_dim = SHARD_AMOUNT * SHARD_AMOUNT
        self.agent = DDPGAgent(state_dim, action_dim)

    def tearDown(self):
        self.env.close()

    def test_agent_action_selection(self):
        # 重置环境和Agent
        state = self.env.reset()
        self.agent.reset_noise()

        # 选择动作
        action = self.agent.select_action(state, self.env.accounts_info, noise=False)
        self.assertEqual(len(action), SHARD_AMOUNT * SHARD_AMOUNT)
        self.assertTrue(np.all(action >= 0.0) and np.all(action <= 1.0), "Action values should be within [0,1]")

    def test_agent_training_step(self):
        # 模拟存储和更新
        state = self.env.reset()
        action = np.zeros(SHARD_AMOUNT * SHARD_AMOUNT, dtype=np.float32)
        reward = 0.0
        next_state = state.copy()
        done = False

        # 存储转移
        self.agent.store_transition(state, action, reward, next_state, done)
        self.agent.update()  # 即使只有一个样本也不会报错

        # 添加足够的样本以触发更新
        for _ in range(BATCH_SIZE):
            self.agent.store_transition(state, action, reward, next_state, done)
        self.agent.update()  # 现在应该执行梯度更新

        # 无断言，仅确保不报错

    def test_agent_training_loop(self):
        # 运行一个小规模的训练循环
        n_steps = 100
        state = self.env.reset()
        self.agent.reset_noise()
        for _ in range(n_steps):
            action = self.agent.select_action(state, self.env.accounts_info, noise=False)
            # 创建虚拟的下一个状态和奖励
            next_state = state.copy()
            reward = 1.0
            done = False
            self.agent.store_transition(state, action, reward, next_state, done)
            self.agent.update()
            state = next_state

        # 无断言，仅确保不报错


if __name__ == "__main__":
    unittest.main()
