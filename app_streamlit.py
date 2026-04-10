import streamlit as st
import pandas as pd
import numpy as np
from algorithms import KillAlgorithms
from rotation_manager import RotationManager
import base64

st.set_page_config(page_title="排列五智能杀号系统", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = None

st.title("🔢 排列五智能杀号预测系统")
st.markdown("基于8种数学算法的完整杀号矩阵 | 上期错误方法本期参考")

# 侧边栏
with st.sidebar:
    st.header("📊 数据配置")
    uploaded_file = st.file_uploader("上传历史数据(Excel)", type=['xlsx', 'xls'])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = [str(col).strip().replace(' ', '') for col in df.columns]
            st.session_state.history = df
            st.success(f"已加载 {len(df)} 期数据")
        except Exception as e:
            st.error(f"读取文件失败: {str(e)}")
    
    if st.session_state.history is not None:
        df = st.session_state.history
        
        st.markdown("---")
        st.header("🎯 期号选择")
        
        if '开奖期号' not in df.columns:
            st.error("数据中未找到'开奖期号'列，请检查Excel格式")
            analysis_df = df.copy()
            selected_period = "未知期号"
            show_result = False
        else:
            df = df.sort_values('开奖期号').reset_index(drop=True)
            period_list = df['开奖期号'].astype(str).tolist()
            
            selected_period = st.selectbox(
                "选择要预测的期号（基于该期之前的历史数据预测）",
                period_list,
                index=len(period_list)-1,
                help="选择后，系统将使用该期之前的所有历史数据进行预测（不包含该期本身）"
            )
            
            selected_idx = df[df['开奖期号'].astype(str) == selected_period].index[0]
            
            st.markdown("---")
            st.header("⚙️ 预测设置")
            base_periods = st.slider("基于前面多少期数据计算", 30, 200, 100)
            
            if selected_idx < base_periods:
                start_idx = 0
                actual_used_periods = selected_idx
            else:
                start_idx = selected_idx - base_periods
                actual_used_periods = base_periods
            
            start_period = df.iloc[start_idx]['开奖期号']
            end_period = df.iloc[selected_idx - 1]['开奖期号'] if selected_idx > 0 else "无"
            
            analysis_df = df.iloc[start_idx:selected_idx].copy()
            
            st.info(f"使用期号 **{start_period}** 至 **{end_period}**（共{actual_used_periods}期）\n\n预测 **{selected_period