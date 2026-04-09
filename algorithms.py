import pandas as pd
import numpy as np
from collections import Counter

class KillAlgorithms:
    def __init__(self, history_df):
        # 清理列名
        self.df = history_df.copy()
        self.df.columns = [str(col).strip().replace(' ', '') for col in self.df.columns]
        
        # 动态检测位置（兼容4位数据）
        possible_positions = ['万位', '千位', '百位', '十位', '个位']
        self.positions = [pos for pos in possible_positions if pos in self.df.columns]
        
        if not self.positions:
            raise ValueError(f"未找到有效的位置列。当前列名: {list(self.df.columns)}")
        
        # 确保数据排序正确
        self.df = self.df.reset_index(drop=True)
        self.latest = self.df.iloc[-1]
        self.recent_10 = self.df.tail(10)
    
    # 安全计算遗漏值
    def _get_omission(self, position, num, end_idx):
        subset = self.df.iloc[:end_idx+1]
        matches = subset[subset[position] == num]
        if matches.empty:
            return 999
        return end_idx - matches.index[-1]

    def method_1_cold_max(self, position):
        """方法1: 极冷号法"""
        curr_idx = len(self.df) - 1
        omissions = {num: self._get_omission(position, num, curr_idx) for num in range(10)}
        return max(omissions, key=omissions.get)
    
    def method_2_hot_decay(self, position):
        """方法2: 热号衰减法"""
        recent_vals = self.recent_10[position].values
        freq = Counter([int(x) for x in recent_vals])
        return max(freq, key=freq.get) if freq else 0
    
    def method_3_markov_low(self, position):
        """方法3: 马尔可夫低概率"""
        if len(self.df) < 2:
            return 0
        last_num = int(self.latest[position])
        transitions = {i: 0 for i in range(10)}
        
        for i in range(len(self.df)-1):
            if int(self.df.iloc[i][position]) == last_num:
                next_num = int(self.df.iloc[i+1][position])
                transitions[next_num] += 1
        
        # 找转移最少的
        min_val = min(transitions.values())
        candidates = [k for k, v in transitions.items() if v == min_val]
        return candidates[0]
    
    def method_4_parity_balance(self, position):
        """方法4: 奇偶纠偏"""
        recent_vals = [int(x) for x in self.recent_10[position].values]
        odd_nums = [x for x in recent_vals if x % 2 == 1]
        even_nums = [x for x in recent_vals if x % 2 == 0]
        
        if len(odd_nums) > len(even_nums):
            freq = Counter(odd_nums)
            return min(freq, key=freq.get) if freq else 1
        else:
            freq = Counter(even_nums)
            return min(freq, key=freq.get) if freq else 0
    
    def method_5_size_deviation(self, position):
        """方法5: 大小偏离"""
        recent_vals = [int(x) for x in self.recent_10[position].values]
        big_nums = [x for x in recent_vals if x >= 5]
        small_nums = [x for x in recent_vals if x < 5]
        
        if len(big_nums) > len(small_nums):
            freq = Counter(big_nums)
            return max(freq, key=freq.get) if freq else 9
        else:
            freq = Counter(small_nums)
            return max(freq, key=freq.get) if freq else 0
    
    def method_6_amplitude_block(self, position):
        """方法6: 振幅阻断"""
        last_val = int(self.latest[position])
        distances = [(i, abs(i - last_val)) for i in range(10)]
        distances.sort(key=lambda x: x[1], reverse=True)
        return distances[0][0]
    
    def method_7_cross_correlation(self, position):
        """方法7: 跨位相关"""
        pos_idx = self.positions.index(position)
        if pos_idx == 0 and len(self.positions) > 1:
            ref_val = int(self.latest[self.positions[1]])
        elif pos_idx > 0:
            ref_val = int(self.latest[self.positions[pos_idx-1]])
        else:
            return 0
        return ref_val  # 杀对子号
    
    def method_8_omission_gradient(self, position):
        """方法8: 遗漏梯度"""
        if len(self.df) < 3:
            return 0
        curr_idx = len(self.df) - 1
        prev_idx = len(self.df) - 2
        
        gradients = {}
        for num in range(10):
            curr_omit = self._get_omission(position, num, curr_idx)
            prev_omit = self._get_omission(position, num, prev_idx)
            gradients[num] = curr_omit - prev_omit
        
        return max(gradients, key=gradients.get)

    def get_all_predictions(self):
        """生成所有预测结果 - 返回字典格式"""
        # 确保方法名与 rotation_manager 完全一致
        method_names = [
            '极冷号法', '热号衰减法', '马尔可夫低概率', 
            '奇偶纠偏法', '大小偏离法', '振幅阻断法', 
            '跨位相关法', '遗漏梯度法'
        ]
        
        results = {}
        for pos in self.positions:
            methods = [
                self.method_1_cold_max(pos),
                self.method_2_hot_decay(pos),
                self.method_3_markov_low(pos),
                self.method_4_parity_balance(pos),
                self.method_5_size_deviation(pos),
                self.method_6_amplitude_block(pos),
                self.method_7_cross_correlation(pos),
                self.method_8_omission_gradient(pos)
            ]
            results[pos] = dict(zip(method_names, methods))
        
        return results