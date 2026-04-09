import random
import streamlit as st

class RotationManager:
    def __init__(self):
        self.methods = [
            '极冷号法', '热号衰减法', '马尔可夫低概率', 
            '奇偶纠偏法', '大小偏离法', '振幅阻断法', 
            '跨位相关法', '遗漏梯度法'
        ]
        self.positions = ['万位', '千位', '百位', '十位', '个位']
        
        if 'last_used' not in st.session_state:
            st.session_state.last_used = {}
    
    def select_methods_for_position(self, position, all_preds, avoid_methods=None):
        """
        选择2种方法，确保：
        1. 不与上期重复（如果avoid_methods提供）
        2. 杀号结果不相同（如果相同，替换第二种方法）
        """
        if avoid_methods is None:
            avoid_methods = st.session_state.last_used.get(position, [])
        
        # 获取所有候选方法及其杀号结果
        available = []
        for m in self.methods:
            if m in all_preds.get(position, {}):
                available.append((m, all_preds[position][m]))
        
        # 排除上期使用的方法（如果还有其他选择）
        candidates = [(m, num) for m, num in available if m not in avoid_methods]
        if len(candidates) < 2:
            candidates = available  # 如果排除后不够，使用全部
        
        if len(candidates) < 2:
            # 如果只有1种方法可用，复制一份（理论上不会发生）
            return [candidates[0][0], candidates[0][0]], [candidates[0][1], candidates[0][1]]
        
        # 随机选择第一种方法
        first = random.choice(candidates)
        selected_methods = [first[0]]
        selected_nums = [first[1]]
        
        # 选择第二种方法（确保杀号不同）
        remaining = [(m, num) for m, num in candidates if m != first[0]]
        
        # 优先选择杀号与第一种不同的方法
        different_num_candidates = [(m, num) for m, num in remaining if num != first[1]]
        
        if different_num_candidates:
            second = random.choice(different_num_candidates)
        else:
            # 如果所有剩余方法杀号都相同（概率极低），强制选一个不同的方法，杀号取次优
            second = random.choice(remaining)
            # 如果杀号相同，强制改为另一个号码（0-9中排除first[1]的第一个）
            if second[1] == first[1]:
                forced_num = (first[1] + 1) % 10
                second = (second[0], forced_num)
        
        selected_methods.append(second[0])
        selected_nums.append(second[1])
        
        return selected_methods, selected_nums
    
    def save_state(self, current_selection):
        """保存本期选择供下期参考"""
        st.session_state.last_used = current_selection.copy()
    
    def get_method_pool(self):
        return {
            '极冷号法': '基于最大遗漏值，杀最冷的号码',
            '热号衰减法': '基于近期频率，杀最热号码',
            '马尔可夫低概率': '基于状态转移矩阵，杀转移概率最低的',
            '奇偶纠偏法': '基于奇偶均衡，杀过度代表的奇或偶',
            '大小偏离法': '基于大小号均衡，杀过度代表的一方',
            '振幅阻断法': '基于与上期差值，杀最远距离号码',
            '跨位相关法': '基于相邻位相关性，杀对子号',
            '遗漏梯度法': '基于遗漏值增速，杀加速冷冻的号码'
        }