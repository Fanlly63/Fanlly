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
        self.recent_20 = self.df.tail(20)
    
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
        # 所有具有最大遗漏值的号码中选最大的那个（避免随机）
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
        # 多个热号时选最大的（确定性，非随机）
        return max(hottest)

    def method_3_012_road(self, position):
        """012路排除法（新）：基于除3余数，杀最热路中的最热号"""
        recent_vals = [int(x) for x in self.recent_10[position].values]
        
        # 0路：0369，1路：147，2路：258
        roads = {0: [], 1: [], 2: []}
        for num in recent_vals:
            roads[num % 3].append(num)
        
        # 统计各路出现频次
        road_counts = {r: len(nums) for r, nums in roads.items()}
        hottest_road = max(road_counts, key=road_counts.get)
        
        # 在最热路中，杀出现次数最多的号码（多个时选最大的）
        if roads[hottest_road]:
            road_freq = Counter(roads[hottest_road])
            max_count = max(road_freq.values())
            candidates = [k for k, v in road_freq.items() if v == max_count]
            return max(candidates)
        return hottest_road  # 备用

    def method_4_parity_balance(self, position):
        """奇偶纠偏法：杀过度代表的奇或偶"""
        recent_vals = [int(x) for x in self.recent_10[position].values]
        odd_nums = [x for x in recent_vals if x % 2 == 1]
        even_nums = [x for x in recent_vals if x % 2 == 0]
        
        if len(odd_nums) > len(even_nums):
            freq = Counter(odd_nums)
            if freq:
                # 选频率最小的奇数（多个时选最小的号码，避免0优先）
                min_count = min(freq.values())
                candidates = [k for k, v in freq.items() if v == min_count]
                return min(candidates)
            return 1
        else:
            freq = Counter(even_nums)
            if freq:
                min_count = min(freq.values())
                candidates = [k for k, v in freq.items() if v == min_count]
                # 偶数中如果0是候选且还有其他，优先选非0的较小值
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
                return max(candidates)  # 杀最热的大号
            return 9
        else:
            freq = Counter(small_nums)
            if freq:
                max_count = max(freq.values())
                candidates = [k for k, v in freq.items() if v == max_count]
                # 小数中如果0是候选且还有其他，优先选非0的较大值（避免0聚集）
                if 0 in candidates and len(candidates) > 1:
                    candidates.remove(0)
                    return max(candidates)
                return max(candidates)
            return 0

    def method_6_amplitude_block(self, position):
        """振幅阻断法（修复为确定性）：距离上期最远的号码"""
        last_val = int(self.latest[position])
        # 计算距离
        distances = [(i, abs(i - last_val)) for i in range(10)]
        max_dist = max([d[1] for d in distances])
        # 所有最大距离中选号码最大的（确定性）
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
        
        # 杀对子（同号），如果0就对子杀0，否则杀对子号
        return ref_val

    def method_8_slope_block(self, position):
        """斜连号阻断法（新，替换原遗漏梯度）：杀上期±1中的热号"""
        last_val = int(self.latest[position])
        
        # 上期号码的左右邻居（循环：9+1=0，0-1=9）
        left = (last_val - 1) % 10  # 左斜连
        right = (last_val + 1) % 10  # 右斜连
        
        # 统计这两个斜连号在近期的出现频率
        recent_vals = [int(x) for x in self.recent_10[position].values]
        freq = Counter(recent_vals)
        
        left_count = freq.get(left, 0)
        right_count = freq.get(right, 0)
        
        # 杀掉较热的那个斜连号（如果一样热，杀号码大的，确定性）
        if left_count > right_count:
            return left
        elif right_count > left_count:
            return right
        else:
            # 频率相同，杀号码大的（避免默认杀0）
            return max(left, right)

    def get_all_predictions(self):
        """生成所有预测结果 - 8种方法（删除随机，新增012路和斜连号）"""
        method_names = [
            '极冷号法', '热号衰减法', '012路排除法',  # 012路替换原马尔可夫
            '奇偶纠偏法', '大小偏离法', '振幅阻断法', 
            '跨位相关法', '斜连号阻断法'  # 斜连号替换原遗漏梯度
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