import csv
import hashlib
from collections import deque
import numpy as np
import random

import config
from config import (
    SHARD_AMOUNT, STATE_WINDOW, DATA_FILE, SEED, ACTIVATE_THRESHOLD,
    MIGRATION_COST, INCENTIVE, PENALTY_COEFF, REWARD_DELAY
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

        # 账户信息数据结构:
        self.accounts_info = {}

        # 分配新账户计数器
        self.n = 0

        # 当前块编号
        self.current_block_number = 0

        # 记录上一次发起迁移提案的块编号
        self.last_proposal_block = -999999999

        # 上一状态的跨片交易总数，用于计算improvement
        self.prev_cross_shard_sum = self._compute_cross_shard_sum_init()

        # 延迟动作队列: 存储 (action_block, action, cross_shard_sum_before)
        self.pending_actions = []

        self.reset()

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

        self.current_block_number = 0
        self.n = 0
        self.last_proposal_block = -999999999
        self.pending_actions.clear()
        self.prev_cross_shard_sum = self._compute_cross_shard_sum_init()
        for i in range(config.STATE_WINDOW):
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

    def _assign_shards(self, from_acc, to_acc):
        from_known = (from_acc in self.accounts_info)
        to_known = (to_acc in self.accounts_info)

        if not from_known and not to_known:
            shard_f = self._assign_new_shard()
            shard_t = self._assign_new_shard()
            self.accounts_info[from_acc] = {
                "shard": shard_f,
                "last_active_block": self.current_block_number,
                "cross_shard_count": 0,
                "intra_shard_count": 0,
                "cross_shard_distribution": {}
            }
            self.accounts_info[to_acc] = {
                "shard": shard_t,
                "last_active_block": self.current_block_number,
                "cross_shard_count": 0,
                "intra_shard_count": 0,
                "cross_shard_distribution": {}
            }
        elif from_known and not to_known:
            shard_f = self.accounts_info[from_acc]["shard"]
            self.accounts_info[to_acc] = {
                "shard": shard_f,
                "last_active_block": self.current_block_number,
                "cross_shard_count": 0,
                "intra_shard_count": 0,
                "cross_shard_distribution": {}
            }
        elif to_known and not from_known:
            shard_t = self.accounts_info[to_acc]["shard"]
            self.accounts_info[from_acc] = {
                "shard": shard_t,
                "last_active_block": self.current_block_number,
                "cross_shard_count": 0,
                "intra_shard_count": 0,
                "cross_shard_distribution": {}
            }
        else:
            self.accounts_info[from_acc]["last_active_block"] = self.current_block_number
            self.accounts_info[to_acc]["last_active_block"] = self.current_block_number

    def _update_accounts_info(self, from_acc, to_acc, from_shard, to_shard, value=1):
        self.accounts_info[from_acc]["last_active_block"] = self.current_block_number
        self.accounts_info[to_acc]["last_active_block"] = self.current_block_number

        if from_shard == to_shard:
            self.accounts_info[from_acc]["intra_shard_count"] += value
            self.accounts_info[to_acc]["intra_shard_count"] += value
        else:
            self.accounts_info[from_acc]["cross_shard_count"] += value
            self.accounts_info[to_acc]["cross_shard_count"] += value

            dist_f = self.accounts_info[from_acc]["cross_shard_distribution"]
            dist_f[to_shard] = dist_f.get(to_shard, 0) + value

            dist_t = self.accounts_info[to_acc]["cross_shard_distribution"]
            dist_t[from_shard] = dist_t.get(from_shard, 0) + value

    def _clean_inactive_accounts(self):
        to_remove = []
        for acc, info in self.accounts_info.items():
            if self.current_block_number - info["last_active_block"] > self.activate_threshold:
                to_remove.append(acc)
        for acc in to_remove:
            del self.accounts_info[acc]

    def _compute_cross_shard_sum_init(self):
        # 初始化时全为零矩阵，返回0.0即可
        return 0.0

    def _compute_cross_shard_sum(self):
        sum_matrix = np.zeros((self.k, self.k), dtype=np.float32)
        for dist in self.last_distributions:
            sum_matrix += dist
        cross_shard_sum = 0.0
        for i in range(self.k):
            for j in range(self.k):
                if i != j:
                    cross_shard_sum += sum_matrix[i, j]
        return cross_shard_sum

    def _schedule_action_reward(self, action):
        # 当执行动作时，记录当前block和当时的cross_shard_sum
        cross_shard_sum_before = self._compute_cross_shard_sum()
        self.pending_actions.append((self.current_block_number, action, cross_shard_sum_before))

    def _compute_delayed_reward(self):
        # 检查pending_actions中是否有到期的动作
        # reward_delay后计算奖励 = 改善 - 成本 - 惩罚 + 激励
        ready_actions = []
        remaining_actions = []
        current = self.current_block_number

        current_cross_shard_sum = self._compute_cross_shard_sum()

        for (action_block, action, cross_shard_sum_before) in self.pending_actions:
            if current - action_block >= self.reward_delay:
                # 到期计算reward
                improvement = cross_shard_sum_before - current_cross_shard_sum

                if len(action) == 0:
                    # 无动作，只是看看改善
                    reward = improvement
                else:
                    cost = MIGRATION_COST * len(action)
                    interval = (action_block - self.last_proposal_block)
                    if interval <= 0:
                        interval = 1
                    penalty = PENALTY_COEFF * (1.0 / interval)
                    incentive = INCENTIVE * len(action)
                    reward = improvement - cost - penalty + incentive

                    self.last_proposal_block = action_block

                ready_actions.append(reward)
            else:
                remaining_actions.append((action_block, action, cross_shard_sum_before))

        self.pending_actions = remaining_actions

        # 若有多个到期动作，我们返回它们的和或只返回最新一个
        # 为简单起见，我们假设一次step只处理一个到期动作（取第一个）
        if len(ready_actions) > 0:
            return ready_actions[0]
        else:
            return 0.0

    def step(self, action):
        # 禁止reward_delay期间发起迁移已在上层逻辑中可实现，这里不强制
        # 执行迁移动作(如果有)
        for (acc, new_shard) in action:
            if acc in self.accounts_info:
                self.accounts_info[acc]["shard"] = new_shard

        if len(action) > 0:
            self._schedule_action_reward(action)

        try:
            row = next(self.reader)
        except StopIteration:
            done = True
            # 没有新块后，计算一次延迟奖励
            delayed_reward = self._compute_delayed_reward()
            return self._build_state(), delayed_reward, done, {}

        self.current_block_number += 1

        from_acc = row.get('from', 'unknown_from')
        to_acc = row.get('to', 'unknown_to')

        self._assign_shards(from_acc, to_acc)
        from_shard = self.accounts_info[from_acc]["shard"]
        to_shard = self.accounts_info[to_acc]["shard"]

        current_distribution = np.zeros((self.k, self.k), dtype=np.float32)
        current_distribution[from_shard, to_shard] += 1.0
        self.last_distributions.append(current_distribution)

        self._update_accounts_info(from_acc, to_acc, from_shard, to_shard, value=1)
        self._clean_inactive_accounts()

        delayed_reward = self._compute_delayed_reward()

        next_state = self._build_state()
        done = False
        info = {}

        return next_state, delayed_reward, done, info
