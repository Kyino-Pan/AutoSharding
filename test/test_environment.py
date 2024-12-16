# test/test_environment.py
import unittest
import os
import numpy as np

from config import SHARD_AMOUNT, DATA_FILE
from env.environment import EthEnvironment

class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.exists(DATA_FILE), f"Data file {DATA_FILE} not found.")
        self.env = EthEnvironment()

    def tearDown(self):
        self.env.close()

    def test_n_blocks_distribution_and_state(self):
        n = 128
        action = []
        steps = 0
        done = False
        last_state = None

        while steps < n and not done:
            next_state, _, done, _ = self.env.step(action)
            last_state = next_state
            steps += 1

        print(f"Processed {steps} blocks.")

        # 打印每个分片账户数量
        acc_count = [0] * SHARD_AMOUNT
        for _, info in self.env.accounts_info.items():
            acc_count[info["shard"]] += 1

        print("Shard account distribution after processing:")
        for i, count in enumerate(acc_count):
            print(f"Shard {i}: {count} accounts")

        if last_state is not None and steps > 0:
            k = SHARD_AMOUNT
            state_matrix = last_state.reshape((k, k))
            print("State matrix after processing n-th block (full print):")
            for row in state_matrix:
                row_str = ' '.join(str(x) for x in row)
                print(row_str)

        # 简单断言：只要不是零步就要求有账户出现（根据实际数据情况调整或移除）
        self.assertTrue(sum(acc_count) > 0 or steps == 0, "No accounts assigned after processing blocks.")


if __name__ == '__main__':
    unittest.main()
