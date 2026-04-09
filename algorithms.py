import pandas as pd
import numpy as np
from collections import Counter

class KillAlgorithms:
    def __init__(self, history_df):
        self.df = history_df.copy()
        self.positions = ['万位', '千位', '百位', '十位', '个位']
        self.latest = self.df.iloc[-1]
        self.recent_10 = self.df.tail(10)
        self.recent_20 = self.df.tail(20)
    
    # 方法1: 极冷号法 - 杀遗漏值最大的
    def method_cold_max(self, position):
        # 从最新期遗漏值数据中提取
        omit_vals = self.latest[f'{position}_遗漏值'] if f'{position}_遗漏值' in self.latest else None
        if omit_vals is None:
            # 计算实时遗漏
            last_appear = {}
            for num in range(10):
                mask = self.df[position] == num
                if mask.any():
                    last_appear[num] = self.df[mask].index[-1]
                else:
                    last_appear[num] = -999
            current_idx = len(self.df) - 1
            omissions = {num: current_idx - idx for num, idx in last_appear.items()}
            return max(omissions, key=omissions.get)
        return max(range(10), key=lambda x: omit_vals[x] if isinstance(omit_vals, (list, np.ndarray)) else 0)
    
    # 方法2: 热号衰减法 - 杀近期出现频率最高的
    def method_hot_decay(self, position):
        recent = self.recent_10[position].values
        freq = Counter(recent)
        if freq:
            return max(freq, key=freq.get)
        return np.random.randint(0, 10)
    
    # 方法3: 马尔可夫转移低概率法
    def method_markov_low(self, position):
        last_num = int(self.latest[position])
        # 构建转移矩阵（简化版：只看下一步）
        transitions = {i: [] for i in range(10)}
        for i in range(len(self.df)-1):
            curr = int(self.df.iloc[i][position])
            next_num = int(self.df.iloc[i+1][position])
            transitions[curr].append(next_num)
        
        # 获取当前号码的转移记录
        next_candidates = transitions.get(last_num, [])
        if not next_candidates:
            return (last_num + 1) % 10
        
        # 找出现最少的（转移概率最低）
        freq = Counter(next_candidates)
        return min(freq, key=freq.get)
    
    # 方法4: 奇偶失衡纠偏法
    def method_parity_balance(self, position):
        recent = self.recent_10[position].values
        odd_count = sum(1 for x in recent if int(x) % 2 == 1)
        even_count = 10 - odd_count
        
        # 如果奇数偏多（>6），杀奇数；反之杀偶数
        if odd_count > 6:
            # 返回近期出现最少的奇数
            odd_nums = [x for x in recent if int(x) % 2 == 1]
            freq = Counter(odd_nums)
            return min(freq, key=freq.get) if freq else 1
        elif even_count > 6:
            even_nums = [x for x in recent if int(x) % 2 == 0]
            freq = Counter(even_nums)
            return min(freq, key=freq.get) if freq else 0
        else:
            # 均衡时杀最小奇数或偶数
            return 1 if odd_count > even_count else 0
    
    # 方法5: 大小区间偏离法
    def method_size_deviation(self, position):
        recent = self.recent_10[position].values
        big_count = sum(1 for x in recent if int(x) >= 5)
        small_count = 10 - big_count
        
        if big_count > 6:
            # 杀大号中刚出的（遗漏小的）
            big_nums = [x for x in recent if int(x) >= 5]
            freq = Counter(big_nums)
            return max(freq, key=freq.get) if freq else 9
        elif small_count > 6:
            small_nums = [x for x in recent if int(x) < 5]
            freq = Counter(small_nums)
            return max(freq, key=freq.get) if freq else 0
        else:
            return 9 if big_count > small_count else 0
    
    # 方法6: 振幅阻断法（与上期差值最大的方向）
    def method_amplitude_block(self, position):
        last_val = int(self.latest[position])
        # 杀掉距离上期号码最远的2个方向取第一个
        candidates = [(i, abs(i - last_val)) for i in range(10)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    # 方法7: 跨位相关规避法（斜连号规避）
    def method_cross_correlation(self, position):
        pos_idx = self.positions.index(position)
        if pos_idx == 0:  # 万位看千位
            ref_pos = '千位'
        else:
            ref_pos = self.positions[pos_idx - 1]
        
        ref_val = int(self.latest[ref_pos])
        # 杀掉与参考位相同的号（对子）和斜连号（±1）
        avoid = {ref_val, (ref_val + 1) % 10, (ref_val - 1) % 10}
        # 选择其中在冷态的号码
        available = list(avoid)
        return available[0] if available else 0
    
    # 方法8: 遗漏值梯度法（增速最快的）
    def method_omission_gradient(self, position):
        if len(self.df) < 3:
            return 0
        # 计算最近两期的遗漏值变化
        curr_omit = {}
        prev_omit = {}
        
        for num in range(10):
            # 当前期遗漏（简化：距离上次出现的期数）
            curr_idx = len(self.df) - 1
            prev_idx = len(self.df) - 2
            
            mask_curr = self.df.iloc[:curr_idx+1][self.df[position] == num]
            mask_prev = self.df.iloc[:prev_idx+1][self.df[position] == num]
            
            curr_omit[num] = curr_idx - mask_curr.index[-1] if len(mask_curr) > 0 else 999
            prev_omit[num] = prev_idx - mask_prev.index[-1] if len(mask_prev) > 0 else 999
        
        # 找遗漏增加最快的（梯度最大）
        gradients = {num: curr_omit[num] - prev_omit[num] for num in range(10)}
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