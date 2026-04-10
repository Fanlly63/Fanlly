import random
import streamlit as st

class RotationManager:
    def __init__(self):
        # 删除随机杀号法，剩8种
        self.methods = [
            '极冷号法', '热号衰减法', '012路排除法',  # 新
            '奇偶纠偏法', '大小偏离法', '振幅阻断法', 
            '跨位相关法', '斜连号阻断法'  # 新
        ]
        self.positions = ['万位', '千位', '百位', '十位', '个位']
        
        if 'last_used' not in st.session_state:
            st.session_state.last_used = {}
    
    def select_methods_for_position(self, position, all_preds, avoid_methods=None):
        """选择2种方法，确保不与上期重复，且杀号不同"""
        if avoid_methods is None:
            avoid_methods = st.session_state.last_used.get(position, [])
        
        available = []
        for m in self.methods:
            if m in all_preds.get(position, {}):
                available.append((m, all_preds[position][m]))
        
        candidates = [(m, num) for m, num in available if m not in avoid_methods]
        if len(candidates) < 2:
            candidates = available
        
        if len(candidates) < 2:
            return [candidates[0][0], candidates[0][0]], [candidates[0][1], candidates[0][1]]
        
        # 随机选择方法组合（这是轮换需求，是设计上的随机，非算法随机）
        # 注意：这里保留random是为了轮换，但算法本身是确定性的
        import random as rd
        rd.shuffle(candidates)
        first = candidates[0]
        selected_methods = [first[0]]
        selected_nums = [first[1]]
        
        # 找第二个，确保杀号不同
        for c in candidates[1:]:
            if c[1] != first[1]:
                selected_methods.append(c[0])
                selected_nums.append(c[1])
                break
        
        # 如果找不到不同的，强制改号
        if len(selected_methods) < 2:
            second = candidates[1]
            forced_num = (first[1] + 1) % 10
            selected_methods.append(second[0])
            selected_nums.append(forced_num)
        
        return selected_methods, selected_nums
    
    def save_state(self, current_selection):
        st.session_state.last_used = current_selection.copy()
    
    def get_method_pool(self):
        return {
            '极冷号法': '遗漏值最大的号码（多个时取大值）',
            '热号衰减法': '近期频率最高的热号（多个时取大值）',
            '012路排除法': '除3余数（0路0369/1路147/2路258），杀最热路中的最热号',  # 新
            '奇偶纠偏法': '杀过度代表的奇或偶',
            '大小偏离法': '杀过度代表的大号(5-9)或小数(0-4)',
            '振幅阻断法': '距离上期号码最远的号（距离相同取大值）',
            '跨位相关法': '杀对子号（与相邻位相同）',
            '斜连号阻断法': '杀上期±1斜连号中较热的那个（防连号）'  # 新
        }