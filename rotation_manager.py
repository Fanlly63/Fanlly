import random
import json
from datetime import datetime
import streamlit as st

class RotationManager:
    def __init__(self):
        self.methods = [
            '极冷号法', '热号衰减法', '马尔可夫低概率', 
            '奇偶纠偏法', '大小偏离法', 
            '振幅阻断法', '跨位相关法', '遗漏梯度法'
        ]
        self.positions = ['万位', '千位', '百位', '十位', '个位']
        
        # Streamlit session state 存储上期使用情况
        if 'last_used' not in st.session_state:
            st.session_state.last_used = {pos: [] for pos in self.positions}
    
    def select_methods_for_position(self, position, available_methods=None):
        """
        为指定位置选择2种方法，确保不与上期重复
        使用策略：优先选择长期未使用的，其次随机
        """
        last_used = st.session_state.last_used.get(position, [])
        
        if available_methods is None:
            available_methods = self.methods.copy()
        
        # 排除上期使用的（如果有其他选择）
        candidates = [m for m in available_methods if m not in last_used]
        
        if len(candidates) < 2:
            # 如果排除后不够2个，允许复用最久未用的
            candidates = available_methods
        
        # 随机选择2种不同的方法
        selected = random.sample(candidates, 2)
        
        # 更新session state（实际应在预测确认后调用save_state）
        return selected
    
    def save_state(self, current_selection):
        """保存本期选择作为下期禁止使用的参考"""
        st.session_state.last_used = current_selection.copy()
    
    def get_method_pool(self):
        """返回8种方法的详细说明"""
        return {
            '极冷号法': '基于最大遗漏值，杀最冷的号码',
            '热号衰减法': '基于近期频率，杀最热号码（回归均值）',
            '马尔可夫低概率': '基于状态转移矩阵，杀转移概率最低的',
            '奇偶纠偏法': '基于奇偶均衡，杀过度代表的奇或偶',
            '大小偏离法': '基于大小号均衡（0-4小，5-9大），杀过度代表的一方',
            '振幅阻断法': '基于与上期差值，杀最远距离号码',
            '跨位相关法': '基于相邻位相关性，杀对子或斜连号',
            '遗漏梯度法': '基于遗漏值增速，杀加速冷冻的号码'
        }