import pandas as pd
import numpy as np
from collections import Counter

class KillAlgorithms:
    def __init__(self, history_df):
        # 清理列名：去除空格，统一格式
        self.df = history_df.copy()
        self.df.columns = [str(col).strip().replace(' ', '') for col in self.df.columns]
        
        # 动态检测可用位置（兼容4位或5位数据）
        possible_positions = ['万位', '千位', '百位', '十位', '个位']
        self.positions = [pos for pos in possible_positions if pos in self.df.columns]
        
        if not self.positions:
            raise ValueError(f"未找到有效的位置列。当前列名: {list(self.df.columns)}")
        
        # 确保数据按开奖期号排序（如果有的话）
        if '开奖期号' in self.df.columns:
            self.df = self.df.sort_values('开奖期号').reset_index(drop=True)
        
        self.latest = self.df.iloc[-1]
        self.recent_10 = self.df.tail(10)
    
    # 辅助方法：安全地计算遗漏值（距离上次出现的期数）
    def _get_omission(self, position, num, end_idx=None):
        """计算指定号码在end_idx期为止的遗漏值"""
        if end_idx is None:
            end_idx = len(self.df) - 1
        
        # 截取到end_idx的数据（包含）
        subset = self.df.iloc[:end_idx+1]
        
        # 查找该号码最后一次出现的位置
        matches = subset[subset[position] == num]
        
        if matches.empty:
            return 999  # 从未出现，视为极大遗漏
        else:
            last_appear_idx = matches.index[-1]
            return end_idx - last_appear_idx

    # 方法1: 极冷号法 - 杀遗漏值最大的
    def method_cold_max(self, position):
        current_idx = len(self.df) - 1
        omissions = {}
        
        for num in range(10):
            omissions[num] = self._get_omission(position, num, current_idx)
        
        return max(omissions, key=omissions.get)
    
    # 方法2: 热号衰减法 - 杀近期出现频率最高的
    def method_hot_decay(self, position):
        recent_vals = self.recent_10[position].values
        freq = Counter(recent_vals)
        if freq:
            # 找出现次数最多的
            max_freq = max(freq.values())
            hottest = [k for k, v in freq.items() if v == max_freq]
            return int(hottest[0])
        return 0
    
    # 方法3: 马尔可夫转移低概率法
    def method_markov_low(self, position):
        if len(self.df) < 2:
            return 0
            
        last_num = int(self.latest[position])
        
        # 统计从last_num转移到各个号码的次数
        transitions = {i: 0 for i in range(10)}
        
        for i in range(len(self.df)-1):
            curr = int(self.df.iloc[i][position])
            if curr == last_num:
                next_num = int(self.df.iloc[i+1][position])
                transitions[next_num] += 1
        
        # 找转移次数最少的（概率最低）
        min_trans = min(transitions.values())
        if min_trans == 0:
            # 如果有从未转移到的号码，优先选这些
            never_trans = [k for k, v in transitions.items() if v == 0]
            return never_trans[0] if never_trans else 0
        
        candidates = [k for k, v in transitions.items() if v == min_trans]
        return candidates[0]
    
    # 方法4: 奇偶失衡纠偏法
    def method_parity_balance(self, position):
        recent_vals = self.recent_10[position].values
        odd_nums = [int(x) for x in recent_vals if int(x) % 2 == 1]
        even_nums = [int(x) for x in recent_vals if int(x) % 2 == 0]
        
        if len(odd_nums) > len(even_nums):
            # 奇数多，杀出现最少的奇数
            if odd_nums:
                freq = Counter(odd_nums)
                return min(freq, key=freq.get)
            return 1
        else:
            # 偶数多，杀出现最少的偶数
            if even_nums:
                freq = Counter(even_nums)
                return min(freq, key=freq.get)
            return 0
    
    # 方法5: 大小区间偏离法
    def method_size_deviation(self, position):
        recent_vals = self.recent_10[position].values
        big_nums = [int(x) for x in recent_vals if int(x) >= 5]
        small_nums = [int(x) for x in recent_vals if int(x) < 5]
        
        if len(big_nums) > len(small_nums):
            # 大号多，杀最热的大号
            if big_nums:
                freq = Counter(big_nums)
                return max(freq, key=freq.get)
            return 9
        else:
            # 小号多，杀最热的小号
            if small_nums:
                freq = Counter(small_nums)
                return max(freq, key=freq.get)
            return 0
    
    # 方法6: 振幅阻断法
    def method_amplitude_block(self, position):
        last_val = int(self.latest[position])
        # 找距离last_val最远的号码
        distances = [(i, abs(i - last_val)) for i in range(10)]
        distances.sort(key=lambda x: x[1], reverse=True)
        return distances[0][0]
    
    # 方法7: 跨位相关规避法
    def method_cross_correlation(self, position):
        pos_idx = self.positions.index(position)
        
        if pos_idx == 0:
            # 第一位（万位），参考下一期（千位）
            if len(self.positions) > 1:
                ref_pos = self.positions[1]
                ref_val = int(self.latest[ref_pos])
                # 杀对子和斜连号
                candidates = [ref_val, (ref_val + 1) % 10, (ref_val - 1) % 10]
                return candidates[0]
            return 0
        else:
            # 其他位，参考前一位
            ref_pos = self.positions[pos_idx - 1]
            ref_val = int(self.latest[ref_pos])
            candidates = [ref_val, (ref_val + 1) % 10, (ref_val - 1) % 10]
            return candidates[0]
    
    # 方法8: 遗漏值梯度法（修复版）
    def method_omission_gradient(self, position):
        if len(self.df) < 3:
            return 0
            
        curr_idx = len(self.df) - 1
        prev_idx = len(self.df) - 2
        
        gradients = {}
        
        for num in range(10):
            # 当前遗漏
            curr_omit = self._get_omission(position, num, curr_idx)
            # 上期遗漏
            prev_omit = self._get_omission(position, num, prev_idx)
            
            # 梯度 = 当前遗漏 - 上期遗漏（表示遗漏增加速度）
            gradients[num] = curr_omit - prev_omit
        
        # 杀遗漏增加最快的（加速冷冻的号码）
        return max(gradients, key=gradients.get)

    def get_all_predictions(self):
        """获取所有位置的所有方法预测"""
        results = {}
        for pos in self.positions:
            results[pos] = {
                '极冷号法': self.method_cold_max(pos),
                '热号衰减法': self.method_hot_decay(pos),
                '马尔可夫低概率': self.method_markov_low(pos),
                '奇偶纠偏法': self.method_parity_balance(pos),
                '大小偏离法': self.method_size_deviation(pos),
                '振幅阻断法': self.method_amplitude_block(pos),
                '跨位相关法': self.method_cross_correlation(pos),
                '遗漏梯度法': self.method_omission_gradient(pos)
            }
        return results