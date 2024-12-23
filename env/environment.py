import csv
import hashlib
from collections import deque
import numpy as np
import copy
import random

import config
from config import (
    SHARD_AMOUNT, STATE_WINDOW, DATA_FILE, SEED, ACTIVATE_THRESHOLD,
    MIGRATION_COST, PENALTY_COEFF, REWARD_DELAY, BLOCK_PER_DAY
)


class EthEnvironment:
    def __init__(self):
        random.seed(SEED)
        self.file = open(DATA_FILE, 'r', encoding='utf-8')
        self.reader = csv.DictReader(self.file)
        self.k = SHARD_AMOUNT
        self.window = STATE_WINDOW
        self.activate_threshold = ACTIVATE_THRESHOLD
        self.reward_delay = REWARD_DELAY
        # 用于存储最近window个块的k×k交易分布矩阵
        self.last_distributions = deque(maxlen=self.window)
        self.compare_distributions = deque(maxlen=self.window)
        self.unrewarded_action = []
        self.prev_acc_info = {}
        # 账户信息数据结构:
        self.accounts_info = {}

        # 分配新账户计数器
        self.n = 0

        # 当前块编号
        self.current_block_number = 0
        self.act_at_block_num = 0

        # 记录上一次发起迁移提案的块编号
        self.last_proposal_block_num = -999999999

        # 上一状态的跨片交易总数，用于计算improvement
        self.prev_cross_shard_sum = 0.0

    def close(self):
        self.file.close()

    def reset(self):
        self.file.seek(0)
        self.reader = csv.DictReader(self.file)

        self.accounts_info.clear()
        self.last_distributions.clear()
        for _ in range(self.window):
            zero_matrix = np.zeros((self.k, self.k), dtype=np.float32)
            self.last_distributions.append(zero_matrix)

        self.compare_distributions.clear()
        for _ in range(self.window):
            zero_matrix = np.zeros((self.k, self.k), dtype=np.float32)
            self.compare_distributions.append(zero_matrix)

        self.current_block_number = 0
        self.act_at_block_num = 0
        self.n = 0
        self.prev_cross_shard_sum = 0.0
        self.prev_acc_info = {}
        self.accounts_info = {}
        self.step([])
        self.step([])
        self.last_proposal_block_num = self.current_block_number
        for i in range(self.window):
            self.step([])
        return self._build_state()

    def _build_state(self):
        sum_matrix = np.zeros((self.k, self.k), dtype=np.float32)
        for dist in self.last_distributions:
            sum_matrix += dist
        state = sum_matrix.flatten()
        return state

    def _assign_new_shard(self):
        data = str(self.n).encode('utf-8')
        h = hashlib.sha256(data).hexdigest()
        val = int(h, 16)
        shard = val % self.k
        self.n += 1
        return shard

    def _assign_shards(self, from_acc, to_acc, acc_info, together=False):
        from_known = (from_acc in acc_info)
        to_known = (to_acc in acc_info)
        if together:
            if not from_known and not to_known:
                shard_f = self._assign_new_shard()
                shard_t = self._assign_new_shard()
                acc_info[from_acc] = {
                    "shard": shard_f,
                    "last_active_block": self.current_block_number,
                    "cross_shard_count": 0,
                    "intra_shard_count": 0,
                    "cross_shard_distribution": {}
                }
                acc_info[to_acc] = {
                    "shard": shard_t,
                    "last_active_block": self.current_block_number,
                    "cross_shard_count": 0,
                    "intra_shard_count": 0,
                    "cross_shard_distribution": {}
                }
            elif from_known and not to_known:
                shard_f = acc_info[from_acc]["shard"]
                acc_info[to_acc] = {
                    "shard": shard_f,
                    "last_active_block": self.current_block_number,
                    "cross_shard_count": 0,
                    "intra_shard_count": 0,
                    "cross_shard_distribution": {}
                }
            elif to_known and not from_known:
                shard_t = acc_info[to_acc]["shard"]
                acc_info[from_acc] = {
                    "shard": shard_t,
                    "last_active_block": self.current_block_number,
                    "cross_shard_count": 0,
                    "intra_shard_count": 0,
                    "cross_shard_distribution": {}
                }
            else:
                acc_info[from_acc]["last_active_block"] = self.current_block_number
                acc_info[to_acc]["last_active_block"] = self.current_block_number
        else:
            if not from_known:
                shard_f = self._assign_new_shard()
                acc_info[from_acc] = {
                    "shard": shard_f,
                    "last_active_block": self.current_block_number,
                    "cross_shard_count": 0,
                    "intra_shard_count": 0,
                    "cross_shard_distribution": {}
                }
            if not to_known:
                shard_t = self._assign_new_shard()
                acc_info[to_acc] = {
                    "shard": shard_t,
                    "last_active_block": self.current_block_number,
                    "cross_shard_count": 0,
                    "intra_shard_count": 0,
                    "cross_shard_distribution": {}
                }
        return acc_info[from_acc]["shard"], acc_info[to_acc]["shard"]

    def _update_accounts_info(self, from_acc, to_acc, from_shard, to_shard, acc_info, value=1):
        acc_info[from_acc]["last_active_block"] = self.current_block_number
        acc_info[to_acc]["last_active_block"] = self.current_block_number
        if from_shard == to_shard:
            acc_info[from_acc]["intra_shard_count"] += value
            acc_info[to_acc]["intra_shard_count"] += value
        else:
            acc_info[from_acc]["cross_shard_count"] += value
            acc_info[to_acc]["cross_shard_count"] += value
            dist_f = acc_info[from_acc]["cross_shard_distribution"]
            dist_f[to_shard] = dist_f.get(to_shard, 0) + value

            dist_t = acc_info[to_acc]["cross_shard_distribution"]
            dist_t[from_shard] = dist_t.get(from_shard, 0) + value

    def _compute_tx_dist_(self, distributions):
        sum_matrix = np.zeros((self.k, self.k), dtype=np.float32)
        for dist in distributions:
            sum_matrix += dist
        cross_sum = 0.0
        intra_sum = 0.0
        shards_dist = np.zeros(self.k, dtype=np.integer)
        for i in range(self.k):
            intra_sum += sum_matrix[i, i]
            shards_dist[i] += sum_matrix[i, i]
            for j in range(self.k):
                if i != j:
                    shards_dist[i] += sum_matrix[i, j]
                    shards_dist[j] += sum_matrix[i, j]
                    cross_sum += sum_matrix[i, j]
        return intra_sum, cross_sum, shards_dist

    def _compute_delayed_reward(self):
        # 检查pending_actions中是否有到期的动作
        # reward_delay后计算奖励 = 改善 - 成本 - 惩罚 + 激励
        current_intra, current_cross, current_dist = self._compute_tx_dist_(self.last_distributions)
        current_ratio = current_intra / (current_intra + current_cross)
        prev_intra, prev_cross, prev_dist = self._compute_tx_dist_(self.compare_distributions)
        prev_ratio = prev_intra / (prev_cross + prev_intra)
        # 评估current_dist
        current_cov = np.std(current_dist) / np.mean(current_dist)
        prev_cov = np.std(prev_dist) / np.mean(prev_dist)
        print(f"\tCov Improve = {current_cov - prev_cov} {current_dist}")
        if prev_ratio >= current_ratio:  # 没迁移情况下的跨片交易比例更大
            improvement = (prev_cross - current_cross) / prev_cross
        else:
            improvement = -(prev_intra - current_intra) / prev_intra

        cov_improvement = prev_cov - current_cov
        if cov_improvement > 0:  # 如果cov更小了，说明分布更均匀了
            improvement = improvement * (1 + cov_improvement)
        else:
            improvement = improvement * (1 + 100 * cov_improvement)

        cost = MIGRATION_COST * len(self.unrewarded_action)
        interval = (self.act_at_block_num - self.last_proposal_block_num)
        self.last_proposal_block_num = self.act_at_block_num
        if interval <= 0:
            interval = 1
        penalty = PENALTY_COEFF * (1.0 / interval)  # REWARD_DELAY ?= 1024
        incentive = min(cost, max(0, interval - ACTIVATE_THRESHOLD) / (2 * BLOCK_PER_DAY))  # 鼓励发起迁移proposal

        improvement -= 0.02
        reward = 32 * improvement - cost - penalty + incentive

        print(
            f"\t{reward}({improvement})\t {prev_intra}({prev_ratio * 100}%) ---> {current_intra}({current_ratio * 100}%),"
            f" \tinterval = {interval}")
        # 为简单起见，我们假设一次step只处理一个到期动作（取第一个）
        self.unrewarded_action = []
        self.prev_acc_info = {}

        return reward

    def _read_next_block(self):

        # 读取下一块的全部交易
        try:
            first_row = next(self.reader)
        except StopIteration:
            return [], None

        current_block_num = int(first_row['blockNumber'])
        transactions = [first_row]

        # 连续读取同一blockNumber的交易
        for row in self.reader:
            if int(row['blockNumber']) == current_block_num:
                transactions.append(row)
            else:
                # 读到下一个块了，把reader的指针回退一行（此处需技巧，或在下一轮从头处理）
                # 简便方法：存储这个row下次再处理（可以在类中加个缓存）
                # 为简单起见，可以使用一个缓存行
                self._next_cached_row = row
                break
        else:
            # 如果for循环自然结束，没有break，说明读完文件
            self._next_cached_row = None

        return transactions, current_block_num

    def _read_and_process_block(self, migration):
        # 处理migration
        if migration:
            # print(f"Migration at block {self.current_block_number}")
            self.unrewarded_action = migration
            self.prev_acc_info = copy.deepcopy(self.accounts_info)
            self.compare_distributions = copy.deepcopy(self.last_distributions)
            self.act_at_block_num = self.current_block_number
            for (acc, new_shard) in migration:
                if acc in self.accounts_info:
                    self.accounts_info[acc]["shard"] = new_shard
                else:
                    raise RuntimeError("Attempt to migrate unknown account.")

        # 若有缓存行，说明上次_read_next_block多读了一行
        if hasattr(self, '_next_cached_row') and self._next_cached_row is not None:
            current_block_num = int(self._next_cached_row['blockNumber'])
            block = [self._next_cached_row]
            self._next_cached_row = None
            # 再读剩余同块交易
            for tx in self.reader:
                if int(tx['blockNumber']) == current_block_num:
                    block.append(tx)
                else:
                    self._next_cached_row = tx
                    break
        else:
            block, current_block_num = self._read_next_block()
        if len(block) == 0:
            # 没有新块了
            done = True
            return done, 0.0
        current_distribution = np.zeros((self.k, self.k), dtype=np.float32)
        prev_distribution = np.zeros((self.k, self.k), dtype=np.float32)
        for tx in block:
            from_acc = tx.get('from', 'unknown_from')
            to_acc = tx.get('to', 'unknown_to')

            from_shard, to_shard = self._assign_shards(from_acc, to_acc, self.accounts_info)
            current_distribution[from_shard, to_shard] += 1.0
            self._update_accounts_info(from_acc, to_acc, from_shard, to_shard, self.accounts_info, value=1)
            if self.unrewarded_action:
                from_shard, to_shard = self._assign_shards(from_acc, to_acc, self.prev_acc_info)
                prev_distribution[from_shard][to_shard] += 1.0
                self._update_accounts_info(from_acc, to_acc, from_shard, to_shard, self.prev_acc_info, value=1)

        self.current_block_number = current_block_num

        self.last_distributions.append(current_distribution)

        if self.unrewarded_action:
            self.compare_distributions.append(prev_distribution)

        done = False
        if migration:
            for i in range(REWARD_DELAY):
                _, done = self._read_and_process_block([])
                if done:
                    break
            delayed_reward = self._compute_delayed_reward()
            return done, delayed_reward
        return done, 0.0

    def step(self, migration):
        done, delayed_reward = self._read_and_process_block(migration)
        if done:
            return self._build_state(), delayed_reward, True
        next_state = self._build_state()
        return next_state, delayed_reward, False
