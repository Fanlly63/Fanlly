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
st.markdown("基于10种数学算法的组选杀号策略 | 上期号码决定本期杀号组")

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
            st.error("数据中未找到'开奖期号'列")
            analysis_df = df.copy()
            selected_period = "未知期号"
            show_result = False
        else:
            df = df.sort_values('开奖期号').reset_index(drop=True)
            period_list = df['开奖期号'].astype(str).tolist()
            
            selected_period = st.selectbox(
                "选择要预测的期号",
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
            
            info_text = f"使用期号 {start_period} 至 {end_period}（共{actual_used_periods}期）\n\n预测 {selected_period}期"
            st.info(info_text)
            
            is_latest = (selected_idx == len(df) - 1)
            
            if not is_latest:
                show_result = st.checkbox("显示实际开奖验证", value=False)
            else:
                show_result = False
                st.info("ℹ️ 当前为最新期")
            
            actual_result = None
            if show_result and selected_idx < len(df):
                actual_result = df.iloc[selected_idx]

# 主界面
if st.session_state.history is not None and 'analysis_df' in locals():
    
    try:
        algo = KillAlgorithms(analysis_df)
        rot_manager = RotationManager()
        
        all_preds = algo.get_all_predictions()
        available_positions = list(all_preds.keys())
        
        # 获取上期号码（用于决定组）
        prev_idx = selected_idx - 1
        if prev_idx >= 0:
            prev_period = df.iloc[prev_idx]['开奖期号']
            prev_data = df.iloc[prev_idx]
        else:
            prev_period = "无"
            prev_data = None
        
        col_left, col_right = st.columns([1, 3])
        
        with col_left:
            st.subheader("📈 10种算法（0-9）")
            st.caption("5组配对：0-5, 1-6, 2-7, 3-8, 4-9")
            for name, desc in rot_manager.get_method_pool().items():
                with st.expander(f"{name}"):
                    st.caption(desc)
        
        with col_right:
            st.subheader(f"🎯 {selected_period}期 组选杀号预测")
            
            if prev_data is not None:
                st.caption(f"基于上期({prev_period})号码自动选择杀号组")
                
                # 显示组选择结果
                group_selection = []
                for pos in available_positions:
                    last_num = int(prev_data[pos])
                    group = last_num % 5
                    group_methods = rot_manager.groups[group]
                    kills = [all_preds[pos][m] for m in group_methods]
                    
                    group_selection.append({
                        '位置': pos,
                        '上期号码': last_num,
                        '所属组': f"组{group}",
                        '使用方法': f"方法{group} + 方法{group+5}",
                        '杀号1': kills[0],
                        '杀号2': kills[1],
                        '保留号': [i for i in range(10) if i not in kills]
                    })
                
                group_df = pd.DataFrame(group_selection)
                st.table(group_df)
                
                # 显示10×4完整矩阵（供参考）
                with st.expander("查看全部10种方法矩阵（参考）"):
                    matrix_data = []
                    for method in rot_manager.methods:
                        row = {'方法': method}
                        for pos in available_positions:
                            row[pos] = all_preds[pos][method]
                        matrix_data.append(row)
                    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)
                
                # 验证结果
                if show_result and actual_result is not None:
                    st.markdown("---")
                    st.subheader("✅ 验证结果")
                    
                    verify_data = []
                    for pos in available_positions:
                        actual_num = int(actual_result[pos])
                        last_num = int(prev_data[pos])
                        group = last_num % 5
                        group_methods = rot_manager.groups[group]
                        kills = [all_preds[pos][m] for m in group_methods]
                        
                        is_correct = actual_num not in kills
                        
                        verify_data.append({
                            '位置': pos,
                            '上期': last_num,
                            '组': group,
                            '杀号': kills,
                            '实际': actual_num,
                            '结果': "✅正确" if is_correct else "❌错误"
                        })
                    
                    verify_df = pd.DataFrame(verify_data)
                    st.table(verify_df)
                    
                    correct_count = sum([1 for v in verify_data if v['结果'] == "✅正确"])
                    st.metric("正确率", f"{correct_count}/{len(verify_data)} = {correct_count/len(verify_data):.1%}")
                    
            else:
                st.warning("无上期数据，无法确定杀号组。请选择非第一期进行预测。")
                
    except Exception as e:
        st.error(f"系统运行错误: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
else:
    st.info("👈 请先上传历史数据文件")

st.markdown("---")
st.caption("⚠️ 免责声明：本系统仅供娱乐和数学研究，不构成投注建议。")