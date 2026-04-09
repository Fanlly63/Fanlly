import pandas as pd
import numpy as np
from collections import Counter
import random

class KillAlgorithms:
    def __init__(self, history_df):
        self.df = history_df.copy()
        self.df.columns = [str(col).strip().replace(' ', '') for col in self.df.columns]
        
        possible_positions = ['万位', '千位', '百位', '十位', '个位']
        self.positions = [pos for pos in possible_positions if pos in self.df.columns]
        
        if not self.positions:
            raise ValueError(f"未找到有效的位置列。当前列名: {list(self.df.columns)}")
        
        self.df = self.df.reset_index(drop=True)
        self.latest = self.df.iloc[-1]
        self.recent_10 = self.df.tail(10)
    
    def _get_omission(self, position, num, end_idx):
        subset = self.df.iloc[:end_idx+1]
        matches = subset[subset[position] == num]
        if matches.empty:
            return 999
        return end_idx - matches.index[-1]

    def method_1_cold_max(self, position):
        curr_idx = len(self.df) - 1
        omissions = {num: self._get_omission(position, num, curr_idx) for num in range(10)}
        # 如果有多个最冷，选最大的（避免0优先，因为0是数字起始）
        max_omit = max(omissions.values())
        candidates = [k for k, v in omissions.items() if v == max_omit]
        return max(candidates)  # 选最大的那个，减少0被选概率

    def method_2_hot_decay(self, position):
        recent_vals = self.recent_10[position].values
        freq = Counter([int(x) for x in recent_vals])
        if not freq:
            return 0
        max_freq = max(freq.values())
        hottest = [k for k, v in freq.items() if v == max_freq]
        # 如果有多个最热，随机选，不固定选0
        return int(np.random.choice(hottest))

    def method_3_markov_low(self, position):
        if len(self.df) < 2:
            return random.randint(0, 9)
        last_num = int(self.latest[position])
        transitions = {i: 0 for i in range(10)}
        
        for i in range(len(self.df)-1):
            if int(self.df.iloc[i][position]) == last_num:
                next_num = int(self.df.iloc[i+1][position])
                transitions[next_num] += 1
        
        min_val = min(transitions.values())
        candidates = [k for k, v in transitions.items() if v == min_val]
        # 随机选，避免总是选0
        return int(np.random.choice(candidates))

    def method_4_parity_balance(self, position):
        recent_vals = [int(x) for x in self.recent_10[position].values]
        odd_nums = [x for x in recent_vals if x % 2 == 1]
        even_nums = [x for x in recent_vals if x % 2 == 0]
        
        if len(odd_nums) > len(even_nums):
            freq = Counter(odd_nums)
            return min(freq, key=freq.get) if freq else 1
        else:
            freq = Counter(even_nums)
            # 如果杀偶数且0是候选，优先杀其他偶数（2,4,6,8）
            if freq:
                if 0 in freq and len(freq) > 1:
                    # 暂时移除0，选其他最小频率的偶数
                    non_zero = {k: v for k, v in freq.items() if k != 0}
                    if non_zero:
                        return min(non_zero, key=non_zero.get)
                return min(freq, key=freq.get)
            return 0

    def method_5_size_deviation(self, position):
        recent_vals = [int(x) for x in self.recent_10[position].values]
        big_nums = [x for x in recent_vals if x >= 5]
        small_nums = [x for x in recent_vals if x < 5]
        
        if len(big_nums) > len(small_nums):
            freq = Counter(big_nums)
            return max(freq, key=freq.get) if freq else 9
        else:
            freq = Counter(small_nums)
            # 如果杀小数且0是候选，优先杀其他小数（1,2,3,4）
            if freq:
                if 0 in freq and len(freq) > 1:
                    non_zero = {k: v for k, v in freq.items() if k != 0}
                    if non_zero:
                        return max(non_zero, key=non_zero.get)
                return max(freq, key=freq.get)
            return 0

    def method_6_amplitude_balance(self, position):
        """修复：振幅平衡法，不再总是杀最远（0），而是杀最远或最近（轮流）"""
        last_val = int(self.latest[position])
        distances = [(i, abs(i - last_val)) for i in range(10)]
        
        # 策略：一半时间杀最远（原逻辑），一半时间杀最近（新逻辑）
        # 用期号奇偶决定，确保轮换
        is_even_period = len(self.df) % 2 == 0
        
        if is_even_period:
            # 杀振幅最大（最远）
            distances.sort(key=lambda x: x[1], reverse=True)
            candidates = [d[0] for d in distances if d[1] == distances[0][1]]
        else:
            # 杀振幅最小（最近，但排除本身）
            distances = [(i, abs(i - last_val)) for i in range(10) if i != last_val]
            distances.sort(key=lambda x: x[1])
            candidates = [d[0] for d in distances if d[1] == distances[0][1]]
        
        # 如果候选包含0和其他，50%概率避开0（除非只有0）
        if 0 in candidates and len(candidates) > 1:
            if random.random() > 0.5:
                candidates.remove(0)
        
        return int(np.random.choice(candidates))

    def method_7_cross_correlation(self, position):
        pos_idx = self.positions.index(position)
        if pos_idx == 0 and len(self.positions) > 1:
            ref_val = int(self.latest[self.positions[1]])
        elif pos_idx > 0:
            ref_val = int(self.latest[self.positions[pos_idx-1]])
        else:
            return random.randint(0, 9)
        
        # 杀对子或斜连，但如果结果是0，有50%概率改为杀1（避免0过度集中）
        kill_num = ref_val
        if kill_num == 0 and random.random() > 0.5:
            return 1  # 改为杀1，减少0聚集
        return kill_num

    def method_8_omission_gradient(self, position):
        if len(self.df) < 3:
            return random.randint(0, 9)
        curr_idx = len(self.df) - 1
        prev_idx = len(self.df) - 2
        
        gradients = {}
        for num in range(10):
            curr_omit = self._get_omission(position, num, curr_idx)
            prev_omit = self._get_omission(position, num, prev_idx)
            gradients[num] = curr_omit - prev_omit
        
        max_grad = max(gradients.values())
        candidates = [k for k, v in gradients.items() if v == max_grad]
        
        # 如果0是候选且还有其他，随机选择，不优先0
        return int(np.random.choice(candidates))

    def method_9_random(self, position):
        return random.randint(0, 9)

    def get_all_predictions(self):
        method_names = [
            '极冷号法', '热号衰减法', '马尔可夫低概率', 
            '奇偶纠偏法', '大小偏离法', '振幅平衡法',  # 改名：阻断→平衡
            '跨位相关法', '遗漏梯度法', '随机杀号法'
        ]
        
        results = {}
        for pos in self.positions:
            methods = [
                self.method_1_cold_max(pos),
                self.method_2_hot_decay(pos),
                self.method_3_markov_low(pos),
                self.method_4_parity_balance(pos),
                self.method_5_size_deviation(pos),
                self.method_6_amplitude_balance(pos),  # 新逻辑
                self.method_7_cross_correlation(pos),
                self.method_8_omission_gradient(pos),
                self.method_9_random(pos)
            ]
            results[pos] = dict(zip(method_names, methods))
        
        return results