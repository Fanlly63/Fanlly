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
st.markdown("基于9种数学算法的完整杀号矩阵 | 每位置9种方法独立计算")

# 侧边栏（保持原样）
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
            st.error("数据中未找到'开奖期号'列")
            analysis_df = df.copy()
            selected_period = "未知期号"
            show_result = False
        else:
            df = df.sort_values('开奖期号').reset_index(drop=True)
            period_list = df['开奖期号'].astype(str).tolist()
            
            selected_period = st.selectbox(
                "选择要预测的期号（基于该期之前的历史数据）",
                period_list,
                index=len(period_list)-1
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
            
            st.info(f"使用期号 **{start_period}** 至 **{end_period}**（共{actual_used_periods}期）\n\n预测 **{selected_period}期**")
            
            # 防剧透
            is_latest = (selected_idx == len(df) - 1)
            if not is_latest:
                show_result = st.checkbox("显示实际开奖验证", value=False)
            else:
                show_result = False
                st.info("ℹ️ 当前为最新期")
            
            actual_result = None
            if show_result and selected_idx < len(df):
                actual_result = df.iloc[selected_idx]

# 主界面 - 右边显示9方法矩阵
if st.session_state.history is not None and 'analysis_df' in locals():
    
    try:
        algo = KillAlgorithms(analysis_df)
        rot_manager = RotationManager()
        
        # 获取9种方法的完整预测
        all_preds = algo.get_all_predictions()
        available_positions = list(all_preds.keys())
        method_names = list(list(all_preds.values())[0].keys())  # 获取9种方法名
        
        col_left, col_right = st.columns([1, 3])
        
        with col_left:
            st.subheader("📈 9种算法")
            for name, desc in rot_manager.get_method_pool().items():
                with st.expander(f"{name}"):
                    st.caption(desc)
        
        with col_right:
            st.subheader(f"🎯 {selected_period}期 完整杀号矩阵")
            st.caption(f"基于历史期号 {start_period} 到 {end_period} | 每位置9种方法独立计算")
            
            # 构建9×4矩阵：行=方法，列=位置
            matrix_data = []
            for method in method_names:
                row = {'方法': method}
                for pos in available_positions:
                    row[pos] = all_preds[pos][method]
                matrix_data.append(row)
            
            matrix_df = pd.DataFrame(matrix_data)
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
            
            # 统计每个位置被杀的号码频次（看哪些号码被多种方法同时选中）
            st.markdown("---")
            st.subheader("📊 号码重合度分析（多方法共振杀号）")
            
            for pos in available_positions:
                kill_numbers = [all_preds[pos][m] for m in method_names]
                freq = pd.Series(kill_numbers).value_counts().sort_index()
                
                # 找出被多个方法同时杀的号码（重合度>1）
                duplicates = freq[freq > 1]
                if not duplicates.empty:
                    dup_str = ", ".join([f"{num}号({count}种方法)" for num, count in duplicates.items()])
                    st.text(f"{pos}: {dup_str}")
            
            # 验证结果（如果勾选显示）
            if show_result and actual_result is not None:
                st.markdown("---")
                st.subheader("✅ 验证结果（9方法 vs 实际开奖）")
                
                verify_data = []
                for pos in available_positions:
                    actual_num = int(actual_result[pos])
                    kills = [all_preds[pos][m] for m in method_names]
                    
                    # 统计正确的方法数（没杀中的）
                    correct_methods = [m for m in method_names if all_preds[pos][m] != actual_num]
                    wrong_methods = [m for m in method_names if all_preds[pos][m] == actual_num]
                    
                    verify_data.append({
                        '位置': pos,
                        '实际开奖': actual_num,
                        '9方法杀号': kills,
                        '正确方法数': len(correct_methods),
                        '错误方法': ', '.join(wrong_methods) if wrong_methods else '无'
                    })
                
                verify_df = pd.DataFrame(verify_data)
                st.table(verify_df)
                
                # 总正确率（9方法×4位置=36个预测）
                total_predictions = len(available_positions) * 9
                correct_predictions = sum([v['正确方法数'] for v in verify_data])
                accuracy = correct_predictions / total_predictions
                st.metric("总体正确率", f"{correct_predictions}/{total_predictions} = {accuracy:.1%}")
                
    except Exception as e:
        st.error(f"系统运行错误: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
else:
    st.info("👈 请先上传历史数据文件并选择期号")

st.markdown("---")
st.caption("⚠️ 免责声明：本系统仅供娱乐和数学研究，不构成投注建议。")