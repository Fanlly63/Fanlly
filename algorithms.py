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

    # 方法0: 极冷号法
    def method_0_cold_max(self, position):
        curr_idx = len(self.df) - 1
        omissions = {num: self._get_omission(position, num, curr_idx) for num in range(10)}
        max_omit = max(omissions.values())
        candidates = [k for k, v in omissions.items() if v == max_omit]
        return max(candidates)

    # 方法1: 热号衰减法
    def method_1_hot_decay(self, position):
        recent_vals = [int(x) for x in self.recent_10[position].values]
        freq = Counter(recent_vals)
        if not freq:
            return 0
        max_freq = max(freq.values())
        hottest = [k for k, v in freq.items() if v == max_freq]
        return max(hottest)

    # 方法2: 012路排除法
    def method_2_012_road(self, position):
        recent_vals = [int(x) for x in self.recent_10[position].values]
        total_sum = sum(recent_vals)
        target_road = total_sum % 3
        
        road_numbers = {0: [0, 3, 6, 9], 1: [1, 4, 7], 2: [2, 5, 8]}
        candidates = road_numbers[target_road]
        
        curr_idx = len(self.df) - 1
        omissions = {num: self._get_omission(position, num, curr_idx) for num in candidates}
        max_omit = max(omissions.values())
        coldest = [k for k, v in omissions.items() if v == max_omit]
        return max(coldest)

    # 方法3: 奇偶纠偏法
    def method_3_parity_balance(self, position):
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

    # 方法4: 大小偏离法
    def method_4_size_deviation(self, position):
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

    # 方法5: 振幅阻断法
    def method_5_amplitude_block(self, position):
        last_val = int(self.latest[position])
        distances = [(i, abs(i - last_val)) for i in range(10)]
        max_dist = max([d[1] for d in distances])
        candidates = [i for i, d in distances if d == max_dist]
        return max(candidates)

    # 方法6: 跨位相关法
    def method_6_cross_correlation(self, position):
        pos_idx = self.positions.index(position)
        if pos_idx == 0 and len(self.positions) > 1:
            ref_val = int(self.latest[self.positions[1]])
        elif pos_idx > 0:
            ref_val = int(self.latest[self.positions[pos_idx-1]])
        else:
            return 0
        return ref_val

    # 方法7: 斜连号阻断法
    def method_7_slope_block(self, position):
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

    # 方法8: 对子杀号法（新：杀上期相同号码）
    def method_8_repeat_kill(self, position):
        return int(self.latest[position])

    # 方法9: 遗漏序位杀号法（新：杀与上期在上期时相同遗漏排名的号码）
    def method_9_omission_rank(self, position):
        last_num = int(self.latest[position])
        curr_idx = len(self.df) - 1
        
        # 上期时的遗漏值（到上上期为止）
        prev_idx = curr_idx - 1
        if prev_idx < 0:
            return 0
        
        # 上期时各号码遗漏值及排名
        prev_omissions = {num: self._get_omission(position, num, prev_idx) for num in range(10)}
        prev_sorted = sorted(prev_omissions.items(), key=lambda x: x[1])  # 从小到大（热→冷）
        
        # 找到上期号码在上期时的排名（序位0-9）
        rank = None
        for i, (num, omit) in enumerate(prev_sorted):
            if num == last_num:
                rank = i
                break
        
        if rank is None:
            return 0
        
        # 本期各号码遗漏值，取相同序位（排名）的号码
        curr_omissions = {num: self._get_omission(position, num, curr_idx) for num in range(10)}
        curr_sorted = sorted(curr_omissions.items(), key=lambda x: x[1])
        
        return curr_sorted[rank][0]

    def get_all_predictions(self):
        """生成所有预测结果 - 10种方法（0-9）"""
        method_names = [
            '0-极冷号法', '1-热号衰减法', '2-012路排除法', '3-奇偶纠偏法', '4-大小偏离法',
            '5-振幅阻断法', '6-跨位相关法', '7-斜连号阻断法', '8-对子杀号法', '9-遗漏序位法'
        ]
        
        results = {}
        for pos in self.positions:
            methods = [
                self.method_0_cold_max(pos),
                self.method_1_hot_decay(pos),
                self.method_2_012_road(pos),
                self.method_3_parity_balance(pos),
                self.method_4_size_deviation(pos),
                self.method_5_amplitude_block(pos),
                self.method_6_cross_correlation(pos),
                self.method_7_slope_block(pos),
                self.method_8_repeat_kill(pos),
                self.method_9_omission_rank(pos)
            ]
            results[pos] = dict(zip(method_names, methods))
        
        return results
    
    def get_method_by_group(self, position, last_num):
        """
        根据上期号码返回对应组的两个方法
        组0(0,5), 组1(1,6), 组2(2,7), 组3(3,8), 组4(4,9)
        上期号码X → 使用组(X%5)的两个方法
        """
        group = last_num % 5
        method_indices = [group, group + 5]
        
        all_preds = self.get_all_predictions()
        pos_methods = all_preds[position]
        method_names = list(pos_methods.keys())
        
        selected = {
            f'方法{method_indices[0]}': pos_methods[method_names[method_indices[0]]],
            f'方法{method_indices[1]}': pos_methods[method_names[method_indices[1]]]
        }
        
        return selected, group