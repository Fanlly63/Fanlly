import pandas as pd
import numpy as np
from collections import Counter

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
        """极冷号法：遗漏值最大的号码，多个时选号码值最大的（确定性）"""
        curr_idx = len(self.df) - 1
        omissions = {num: self._get_omission(position, num, curr_idx) for num in range(10)}
        max_omit = max(omissions.values())
        candidates = [k for k, v in omissions.items() if v == max_omit]
        return max(candidates)

    def method_2_hot_decay(self, position):
        """热号衰减法：频率最高的，多个时选最大的号码（确定性）"""
        recent_vals = [int(x) for x in self.recent_10[position].values]
        freq = Counter(recent_vals)
        if not freq:
            return 0
        max_freq = max(freq.values())
        hottest = [k for k, v in freq.items() if v == max_freq]
        return max(hottest)

    def method_3_012_road(self, position):
        """
        012路排除法（新规则）：
        1. 计算前十期号码之和
        2. 和对3取余，确定目标路（0路/1路/2路）
        3. 在该路号码中杀最冷的（遗漏值最大）
        """
        recent_vals = [int(x) for x in self.recent_10[position].values]
        
        # 计算前十期之和的除3余数
        total_sum = sum(recent_vals)
        target_road = total_sum % 3  # 0, 1, 或 2
        
        # 定义各路号码
        road_numbers = {
            0: [0, 3, 6, 9],  # 0路
            1: [1, 4, 7],     # 1路
            2: [2, 5, 8]      # 2路
        }
        
        # 获取目标路的号码列表
        candidates = road_numbers[target_road]
        
        # 在该路号码中找最冷的（遗漏值最大）
        curr_idx = len(self.df) - 1
        omissions = {num: self._get_omission(position, num, curr_idx) for num in candidates}
        
        # 找最大遗漏值，多个时取号码最大的（确定性）
        max_omit = max(omissions.values())
        coldest = [k for k, v in omissions.items() if v == max_omit]
        return max(coldest)

    def method_4_parity_balance(self, position):
        """奇偶纠偏法：杀过度代表的奇或偶"""
        recent_vals = [int(x) for x in self.recent_10[position].values]
        odd_nums = [x for x in recent_vals if x % 2 == 1]
        even_nums = [x for x in recent_vals if x % 2 == 0]
        
        if len(odd_nums) > len(even_nums):
            freq = Counter(odd_nums)
            if freq:
                min_count = min(freq.values())
                candidates = [k for k, v in freq.items() if v == min_count]
                return min(candidates)
            return 1
        else:
            freq = Counter(even_nums)
            if freq:
                min_count = min(freq.values())
                candidates = [k for k, v in freq.items() if v == min_count]
                if 0 in candidates and len(candidates) > 1:
                    candidates.remove(0)
                    return min(candidates)
                return min(candidates)
            return 0

    def method_5_size_deviation(self, position):
        """大小偏离法：杀过度代表的大或小"""
        recent_vals = [int(x) for x in self.recent_10[position].values]
        big_nums = [x for x in recent_vals if x >= 5]
        small_nums = [x for x in recent_vals if x < 5]
        
        if len(big_nums) > len(small_nums):
            freq = Counter(big_nums)
            if freq:
                max_count = max(freq.values())
                candidates = [k for k, v in freq.items() if v == max_count]
                return max(candidates)
            return 9
        else:
            freq = Counter(small_nums)
            if freq:
                max_count = max(freq.values())
                candidates = [k for k, v in freq.items() if v == max_count]
                if 0 in candidates and len(candidates) > 1:
                    candidates.remove(0)
                    return max(candidates)
                return max(candidates)
            return 0

    def method_6_amplitude_block(self, position):
        """振幅阻断法（修复为确定性）：距离上期最远的号码"""
        last_val = int(self.latest[position])
        distances = [(i, abs(i - last_val)) for i in range(10)]
        max_dist = max([d[1] for d in distances])
        candidates = [i for i, d in distances if d == max_dist]
        return max(candidates)

    def method_7_cross_correlation(self, position):
        """跨位相关法（确定性）：杀对子或斜连"""
        pos_idx = self.positions.index(position)
        if pos_idx == 0 and len(self.positions) > 1:
            ref_val = int(self.latest[self.positions[1]])
        elif pos_idx > 0:
            ref_val = int(self.latest[self.positions[pos_idx-1]])
        else:
            return 0
        
        return ref_val

    def method_8_slope_block(self, position):
        """斜连号阻断法（新，替换原遗漏梯度）：杀上期±1中的热号"""
        last_val = int(self.latest[position])
        
        left = (last_val - 1) % 10
        right = (last_val + 1) % 10
        
        recent_vals = [int(x) for x in self.recent_10[position].values]
        freq = Counter(recent_vals)
        
        left_count = freq.get(left, 0)
        right_count = freq.get(right, 0)
        
        if left_count > right_count:
            return left
        elif right_count > left_count:
            return right
        else:
            return max(left, right)

    def get_all_predictions(self):
        """生成所有预测结果 - 8种方法"""
        method_names = [
            '极冷号法', '热号衰减法', '012路排除法',
            '奇偶纠偏法', '大小偏离法', '振幅阻断法', 
            '跨位相关法', '斜连号阻断法'
        ]
        
        results = {}
        for pos in self.positions:
            methods = [
                self.method_1_cold_max(pos),
                self.method_2_hot_decay(pos),
                self.method_3_012_road(pos),
                self.method_4_parity_balance(pos),
                self.method_5_size_deviation(pos),
                self.method_6_amplitude_block(pos),
                self.method_7_cross_correlation(pos),
                self.method_8_slope_block(pos)
            ]
            results[pos] = dict(zip(method_names, methods))
        
        return results