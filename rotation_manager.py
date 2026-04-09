import random
import streamlit as st

class RotationManager:
    def __init__(self):
        # 必须与 algorithms.py 中的 method_names 完全一致
        self.methods = [
            '极冷号法', '热号衰减法', '马尔可夫低概率', 
            '奇偶纠偏法', '大小偏离法', '振幅阻断法', 
            '跨位相关法', '遗漏梯度法'
        ]
        self.positions = ['万位', '千位', '百位', '十位', '个位']
        
        if 'last_used' not in st.session_state:
            st.session_state.last_used = {}
    
    def select_methods_for_position(self, position):
        """为指定位置选择2种方法，确保不与上期重复"""
        # 获取上期该位置使用的方法
        last_used_for_pos = st.session_state.last_used.get(position, [])
        
        # 候选方法（排除上期使用的）
        candidates = [m for m in self.methods if m not in last_used_for_pos]
        
        # 如果排除后不够2个，则使用全部方法
        if len(candidates) < 2:
            candidates = self.methods
        
        # 随机选择2种
        selected = random.sample(candidates, 2)
        return selected
    
    def save_state(self, current_selection):
        """保存本期选择供下期参考"""
        st.session_state.last_used = current_selection.copy()
    
    def get_method_pool(self):
        """返回方法说明"""
        return {
            '极冷号法': '基于最大遗漏值，杀最冷的号码',
            '热号衰减法': '基于近期频率，杀最热号码（回归均值）',
            '马尔可夫低概率': '基于状态转移矩阵，杀转移概率最低的',
            '奇偶纠偏法': '基于奇偶均衡，杀过度代表的奇或偶',
            '大小偏离法': '基于大小号均衡，杀过度代表的一方',
            '振幅阻断法': '基于与上期差值，杀最远距离号码',
            '跨位相关法': '基于相邻位相关性，杀对子号',
            '遗漏梯度法': '基于遗漏值增速，杀加速冷冻的号码'
        }