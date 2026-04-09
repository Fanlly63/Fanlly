import streamlit as st
import pandas as pd
import numpy as np
from algorithms import KillAlgorithms
from rotation_manager import RotationManager
import base64

st.set_page_config(page_title="排列五智能杀号系统", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = None
if 'last_used' not in st.session_state:
    st.session_state.last_used = {}

st.title("🔢 排列五智能杀号预测系统")
st.markdown("基于8种数学算法的轮换杀号策略 | 支持盲测验证")

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
            actual_result = None
            selected_period = "未知期号"
            show_result = False
        else:
            # 排序确保顺序正确
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
            
            # 计算实际数据范围
            if selected_idx < base_periods:
                start_idx = 0
                actual_used_periods = selected_idx
            else:
                start_idx = selected_idx - base_periods
                actual_used_periods = base_periods
            
            # 显示实际期号范围（而非索引号）
            start_period = df.iloc[start_idx]['开奖期号']
            end_period = df.iloc[selected_idx - 1]['开奖期号'] if selected_idx > 0 else "无"
            
            # ⚠️ 关键：截取数据不包含当期（iloc左闭右开）
            analysis_df = df.iloc[start_idx:selected_idx].copy()
            
            st.info(f"使用期号 **{start_period}** 至 **{end_period}**（共{actual_used_periods}期）\n\n预测 **{selected_period}期**（未开奖）")
            
            # 判断是否最新期
            is_latest = (selected_idx == len(df) - 1)
            
            # 防剧透开关
            if not is_latest:
                show_result = st.checkbox("显示实际开奖验证正确率", value=False,
                                        help="盲测模式：先不勾选进行预测，记录后再勾选验证")
            else:
                show_result = False
                st.info("ℹ️ 当前为最新期，实际开奖未知")
            
            # 获取实际开奖（仅在勾选时显示）
            actual_result = None
            if show_result and selected_idx < len(df):
                actual_result = df.iloc[selected_idx]
        
        st.markdown("---")
        prediction_mode = st.radio(
            "预测模式",
            ["自动轮换（防重复）", "手动选择方法"]
        )

# 主界面
if st.session_state.history is not None and 'analysis_df' in locals():
    
    try:
        algo = KillAlgorithms(analysis_df)
        rot_manager = RotationManager()
        
        all_preds = algo.get_all_predictions()
        available_positions = list(all_preds.keys())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"🎯 {selected_period}期 杀号预测")
            st.caption(f"基于历史期号 {start_period} 到 {end_period} 的数据计算 | 共{len(analysis_df)}期")
            
            if prediction_mode == "自动轮换（防重复）":
                current_selection = {}
                final_kill_numbers = {}
                
                for pos in available_positions:
                    try:
                        avoid = st.session_state.last_used.get(pos, [])
                        methods, nums = rot_manager.select_methods_for_position(
                            pos, all_preds, avoid_methods=avoid
                        )
                        
                        current_selection[pos] = methods
                        final_kill_numbers[pos] = nums
                        
                    except Exception as e:
                        st.error(f"处理 {pos} 时出错: {str(e)}")
                        continue
                
                # 显示结果
                if current_selection:
                    result_data = []
                    correct_count = 0
                    total_positions = 0
                    
                    for pos in available_positions:
                        if pos in final_kill_numbers:
                            nums = final_kill_numbers[pos]
                            methods = current_selection[pos]
                            
                            # 验证逻辑（仅在show_result=True时显示实际数据）
                            if actual_result is not None and show_result and pos in actual_result:
                                actual_num = int(actual_result[pos])
                                is_correct = actual_num not in nums
                                total_positions += 1
                                if is_correct:
                                    correct_count += 1
                                status = "✅正确" if is_correct else "❌错误"
                                actual_display = actual_num
                            else:
                                status = "待开奖验证"
                                actual_display = "**隐藏**" if not show_result else "-"
                            
                            safe_nums = [i for i in range(10) if i not in nums]
                            result_data.append({
                                '位置': pos,
                                '杀号1': nums[0],
                                '方法1': methods[0],
                                '杀号2': nums[1],
                                '方法2': methods[1],
                                '保留号': safe_nums,
                                '实际开奖': actual_display,
                                '验证结果': status
                            })
                    
                    if result_data:
                        result_df = pd.DataFrame(result_data)
                        st.table(result_df)
                        
                        # 正确率统计（仅验证模式）
                        if show_result and total_positions > 0:
                            accuracy = correct_count / total_positions
                            st.markdown("---")
                            cols = st.columns(3)
                            cols[0].metric("正确位置", f"{correct_count}/{total_positions}")
                            cols[1].metric("正确率", f"{accuracy:.1%}")
                            if accuracy >= 0.75:
                                cols[2].success("优秀")
                            elif accuracy >= 0.5:
                                cols[2].warning("一般")
                            else:
                                cols[2].error("需优化")
                        
                        # 保存按钮（仅最新期显示）
                        if is_latest:
                            if st.button("✅ 确认本期预测并保存轮换状态", type="primary"):
                                rot_manager.save_state(current_selection)
                                st.success("轮换状态已保存！下期将自动避开本期方法")
                        else:
                            if show_result:
                                st.info("💡 历史期号回测完成，可更换期号继续测试")
            
            else:  # 手动模式
                st.warning("手动模式：请为每个位置选择2种方法")
                
                custom_selection = {}
                for pos in available_positions:
                    with st.expander(f"{pos} 设置"):
                        m1 = st.selectbox(f"{pos} - 方法1", rot_manager.methods, key=f"{pos}_1")
                        remaining = [m for m in rot_manager.methods if m != m1]
                        m2 = st.selectbox(f"{pos} - 方法2", remaining, key=f"{pos}_2")
                        
                        # 检查杀号是否重复
                        if m1 in all_preds[pos] and m2 in all_preds[pos]:
                            n1, n2 = all_preds[pos][m1], all_preds[pos][m2]
                            if n1 == n2:
                                st.error(f"⚠️ 两种方法杀号相同({n1})，请更换！")
                        
                        custom_selection[pos] = [m1, m2]
                
                if st.button("生成手动预测"):
                    manual_data = []
                    for pos in available_positions:
                        if pos in custom_selection:
                            nums = [all_preds[pos][m] for m in custom_selection[pos]]
                            if nums[0] == nums[1]:
                                st.error(f"{pos} 杀号重复，已自动修正")
                                nums[1] = (nums[1] + 1) % 10
                            
                            if actual_result is not None and show_result and pos in actual_result:
                                actual_num = int(actual_result[pos])
                                is_correct = actual_num not in nums
                                status = "✅正确" if is_correct else "❌错误"
                                actual_display = actual_num
                            else:
                                status = "待验证"
                                actual_display = "**隐藏**"
                            
                            manual_data.append({
                                '位置': pos,
                                '杀号1': nums[0],
                                '方法1': custom_selection[pos][0],
                                '杀号2': nums[1],
                                '方法2': custom_selection[pos][1],
                                '实际开奖': actual_display,
                                '验证结果': status
                            })
                    if manual_data:
                        st.table(pd.DataFrame(manual_data))
        
        with col2:
            st.subheader("📈 算法说明")
            for name, desc in rot_manager.get_method_pool().items():
                with st.expander(name):
                    st.write(desc)
            
            st.markdown("---")
            st.subheader("📊 上期回顾")
            if st.session_state.last_used:
                for pos, methods in st.session_state.last_used.items():
                    st.text(f"{pos}: {', '.join(methods)}")
            else:
                st.info("暂无历史记录")
                
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