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
            
            # 简化显示，避免括号冲突
            info_text = f"使用期号 {start_period} 至 {end_period}（共{actual_used_periods}期）\n\n预测 {selected_period}期（未开奖）"
            st.info(info_text)
            
            is_latest = (selected_idx == len(df) - 1)
            
            if not is_latest:
                show_result = st.checkbox("显示实际开奖验证正确率", value=False,
                                        help="盲测模式：先不勾选进行预测，记录后再勾选验证")
            else:
                show_result = False
                st.info("ℹ️ 当前为最新期，实际开奖未知")
            
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
        method_names = list(list(all_preds.values())[0].keys())
        
        col_left, col_right = st.columns([1, 3])
        
        with col_left:
            st.subheader("📈 8种算法")
            for name, desc in rot_manager.get_method_pool().items():
                with st.expander(f"{name}"):
                    st.caption(desc)
        
        with col_right:
            st.subheader(f"🎯 {selected_period}期 完整杀号矩阵")
            st.caption(f"基于历史期号 {start_period} 到 {end_period} | 每位置8种方法独立计算")
            
            matrix_data = []
            for method in method_names:
                row = {'方法': method}
                for pos in available_positions:
                    row[pos] = all_preds[pos][method]
                matrix_data.append(row)
            
            matrix_df = pd.DataFrame(matrix_data)
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
            
            if selected_idx > 0:
                st.markdown("---")
                st.subheader("🔍 上期错误方法参考（均值回归策略）")
                st.caption("上期预测错误的方法，本期可能正确，建议重点关注")
                
                prev_idx = selected_idx - 1
                prev_period = df.iloc[prev_idx]['开奖期号']
                
                if prev_idx < base_periods:
                    prev_start_idx = 0
                else:
                    prev_start_idx = prev_idx - base_periods
                
                prev_analysis_df = df.iloc[prev_start_idx:prev_idx].copy()
                prev_actual = df.iloc[prev_idx]
                
                prev_algo = KillAlgorithms(prev_analysis_df)
                prev_preds = prev_algo.get_all_predictions()
                
                wrong_methods_summary = []
                
                for pos in available_positions:
                    prev_actual_num = int(prev_actual[pos])
                    wrong_methods_list = []
                    
                    for method_name in method_names:
                        prev_kill = prev_preds[pos][method_name]
                        if prev_kill == prev_actual_num:
                            current_kill = all_preds[pos][method_name]
                            wrong_methods_list.append({
                                '方法': method_name,
                                '上期杀号': prev_kill,
                                '本期杀号': current_kill
                            })
                    
                    if wrong_methods_list:
                        wrong_methods_summary.append({
                            '位置': pos,
                            '上期实际': prev_actual_num,
                            '错误方法数': len(wrong_methods_list),
                            '本期建议关注': ' | '.join([f"{m['方法']}杀{m['本期杀号']}" for m in wrong_methods_list])
                        })
                
                if wrong_methods_summary:
                    wrong_df = pd.DataFrame(wrong_methods_summary)
                    st.table(wrong_df)
                    
                    with st.expander("查看详细错误方法对照表"):
                        detail_data = []
                        for pos in available_positions:
                            prev_actual_num = int(prev_actual[pos])
                            for method_name in method_names:
                                if prev_preds[pos][method_name] == prev_actual_num:
                                    detail_data.append({
                                        '位置': pos,
                                        '上期错误方法': method_name,
                                        '上期杀了': prev_preds[pos][method_name],
                                        '上期实际': prev_actual_num,
                                        '本期建议杀': all_preds[pos][method_name]
                                    })
                        if detail_data:
                            st.table(pd.DataFrame(detail_data))
                    
                    total_wrong = sum([d['错误方法数'] for d in wrong_methods_summary])
                    st.info(f"📊 上期({prev_period})共**{total_wrong}个方法**预测错误，本期可重点关注这些方法")
                else:
                    st.success(f"🎉 上期({prev_period})所有8方法全对！本期建议谨慎或反向参考")
            else:
                st.info("ℹ️ 当前选择为期号表第一行，无上期数据可供参考")
            
            if show_result and actual_result is not None:
                st.markdown("---")
                st.subheader("✅ 验证结果（8方法 vs 实际开奖）")
                
                verify_data = []
                for pos in available_positions:
                    actual_num = int(actual_result[pos])
                    kills = [all_preds[pos][m] for m in method_names]
                    
                    correct_methods = [m for m in method_names if all_preds[pos][m] != actual_num]
                    wrong_methods = [m for m in method_names if all_preds[pos][m] == actual_num]
                    
                    verify_data.append({
                        '位置': pos,
                        '实际开奖': actual_num,
                        '8方法杀号': kills,
                        '正确方法数': len(correct_methods),
                        '错误方法': ', '.join(wrong_methods) if wrong_methods else '无'
                    })
                
                verify_df = pd.DataFrame(verify_data)
                st.table(verify_df)
                
                total_predictions = len(available_positions) * 8
                correct_predictions = sum([v['正确方法数'] for v in verify_data])
                accuracy = correct_predictions / total_predictions
                st.metric("总体正确率", f"{correct_predictions}/{total_predictions} = {accuracy:.1%}")
                
    except Exception as e:
        st.error(f"系统运行错误: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
else:
    st.info("👈 请先上传历史数据文件并选择期号")
    
    template = pd.DataFrame(columns=['开奖期号', '万位', '千位', '百位', '十位'])
    csv = template.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="template.csv">下载数据模板</a>'
    st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ 免责声明：本系统仅供娱乐和数学研究，不构成投注建议。")