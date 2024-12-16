import unittest
import os

from config import SHARD_AMOUNT, DATA_FILE
from env.environment import EthEnvironment

class TestEnvironmentAlt(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.exists(DATA_FILE), f"Data file {DATA_FILE} not found.")
        self.env = EthEnvironment()

    def tearDown(self):
        self.env.close()

    def test_128_blocks_unique_accounts(self):
        """
        读取前128个块统计账户出现数量。
        """
        action = 0
        steps = 0
        done = False

        # 在处理块的过程中，account_to_shard字典会记录出现过的账户
        while steps < 128 and not done:
            _, _, done, _ = self.env.step(action)
            steps += 1

        # 已处理steps个块（<=128）
        # 统计出现的账户数量
        total_accounts = len(self.env.account_to_shard)

        print(f"After reading {steps} blocks, {total_accounts} unique accounts appeared.")

        # 根据实际情况断言，如至少有1个账户
        self.assertTrue(total_accounts > 0, "No accounts appeared in first 128 blocks.")


if __name__ == '__main__':
    unittest.main()
