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
st.markdown("基于8种数学算法的轮换杀号策略 | 支持任意期号回测验证")

# 侧边栏
with st.sidebar:
    st.header("📊 数据配置")
    uploaded_file = st.file_uploader("上传历史数据(Excel)", type=['xlsx', 'xls'])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            # 清理列名
            df.columns = [str(col).strip().replace(' ', '') for col in df.columns]
            st.session_state.history = df
            st.success(f"已加载 {len(df)} 期数据")
        except Exception as e:
            st.error(f"读取文件失败: {str(e)}")
    
    if st.session_state.history is not None:
        df = st.session_state.history
        
        # 期号选择功能
        st.markdown("---")
        st.header("🎯 期号选择")
        
        if '开奖期号' in df.columns:
            # 获取所有期号列表
            period_list = df['开奖期号'].astype(str).tolist()
            selected_period = st.selectbox(
                "选择预测期号（基于该期之前的历史数据预测该期）",
                period_list,
                index=len(period_list)-1,
                help="选择要预测的期号，系统将基于该期之前的数据进行预测，并与该期实际开奖对比验证正确率"
            )
            
            # 获取该期在数据中的索引
            selected_idx = df[df['开奖期号'].astype(str) == selected_period].index[0]
            
            # 基础期数设置（用于计算历史数据量）
            st.markdown("---")
            st.header("⚙️ 预测设置")
            base_periods = st.slider("基于前面多少期数据计算", 30, 200, 100, 
                                   help="使用所选期号之前的N期数据进行计算")
            
            # 确定实际可用的数据范围
            if selected_idx < base_periods:
                st.warning(f"所选期号前只有{selected_idx}期数据，将使用全部可用历史")
                start_idx = 0
            else:
                start_idx = selected_idx - base_periods
            
            # 截取数据（不包含当期，只包含历史）
            analysis_df = df.iloc[start_idx:selected_idx].copy()
            actual_result = df.iloc[selected_idx] if selected_idx < len(df) else None
            
            st.info(f"将基于第 {start_idx+1} 期到第 {selected_idx} 期（共{len(analysis_df)}期）预测 **{selected_period}期**")
            
            if actual_result is not None:
                st.success(f"该期实际开奖: 万{actual_result['万位']}千{actual_result['千位']}百{actual_result['百位']}十{actual_result['十位']}")
        else:
            st.error("数据中未找到'开奖期号'列")
            analysis_df = df.copy()
            actual_result = None
            selected_period = "未知期号"
    
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
        
        # 获取所有预测（8种方法的结果）
        all_preds = algo.get_all_predictions()
        available_positions = list(all_preds.keys())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"🎯 {selected_period}期 杀号预测")
            
            if prediction_mode == "自动轮换（防重复）":
                current_selection = {}
                final_kill_numbers = {}
                
                for pos in available_positions:
                    try:
                        # 传入上期使用的方法进行避让
                        avoid = st.session_state.last_used.get(pos, [])
                        methods, nums = rot_manager.select_methods_for_position(
                            pos, all_preds, avoid_methods=avoid
                        )
                        
                        current_selection[pos] = methods
                        final_kill_numbers[pos] = nums
                        
                    except Exception as e:
                        st.error(f"处理 {pos} 时出错: {str(e)}")
                        continue
                
                # 显示结果表格
                if current_selection:
                    result_data = []
                    correct_count = 0
                    total_positions = 0
                    
                    for pos in available_positions:
                        if pos in final_kill_numbers:
                            nums = final_kill_numbers[pos]
                            methods = current_selection[pos]
                            
                            # 验证正确性（如果已知实际开奖）
                            is_correct = "待开奖"
                            if actual_result is not None and pos in actual_result:
                                actual_num = int(actual_result[pos])
                                is_correct = actual_num not in nums
                                total_positions += 1
                                if is_correct:
                                    correct_count += 1
                                status = "✅正确" if is_correct else "❌错误"
                            else:
                                status = "待验证"
                            
                            safe_nums = [i for i in range(10) if i not in nums]
                            result_data.append({
                                '位置': pos,
                                '杀号1': nums[0],
                                '方法1': methods[0],
                                '杀号2': nums[1],
                                '方法2': methods[1],
                                '保留号': safe_nums,
                                '实际开奖': int(actual_result[pos]) if actual_result is not None and pos in actual_result else '-',
                                '验证结果': status
                            })
                    
                    if result_data:
                        result_df = pd.DataFrame(result_data)
                        st.table(result_df)
                        
                        # 显示正确率统计
                        if total_positions > 0:
                            accuracy = correct_count / total_positions
                            st.markdown("---")
                            if accuracy >= 0.8:
                                st.success(f"🎉 本期杀号正确率: {correct_count}/{total_positions} = {accuracy:.1%}")
                            elif accuracy >= 0.5:
                                st.warning(f"⚠️ 本期杀号正确率: {correct_count}/{total_positions} = {accuracy:.1%}")
                            else:
                                st.error(f"❌ 本期杀号正确率: {correct_count}/{total_positions} = {accuracy:.1%}")
                        
                        # 保存按钮（只有在预测最新期时才显示保存）
                        if selected_idx == len(df) - 1:
                            if st.button("✅ 确认本期预测并保存轮换状态", type="primary"):
                                rot_manager.save_state(current_selection)
                                st.success("轮换状态已保存！下期将自动避开本期使用的方法")
                        else:
                            st.info("💡 历史期号回测模式：仅验证正确率，不保存轮换状态")
            
            else:  # 手动模式
                st.warning("手动模式：请为每个位置选择2种不同的方法（确保杀号不同）")
                
                custom_selection = {}
                for pos in available_positions:
                    with st.expander(f"{pos} 设置"):
                        m1 = st.selectbox(f"{pos} - 杀号1", rot_manager.methods, key=f"{pos}_1")
                        # 根据第一种方法的结果，禁用会导致重复杀号的第二种方法
                        first_kill = all_preds[pos][m1]
                        
                        remaining_methods = [m for m in rot_manager.methods if m != m1]
                        m2 = st.selectbox(f"{pos} - 杀号2（杀号: {all_preds[pos][m2] if 'm2' in locals() else '待选'}）", 
                                        remaining_methods, key=f"{pos}_2")
                        
                        # 检查杀号是否相同
                        second_kill = all_preds[pos][m2]
                        if first_kill == second_kill:
                            st.error(f"⚠️ 警告：两种方法杀号相同({first_kill})，请更换第二种方法！")
                        
                        custom_selection[pos] = [m1, m2]
                
                if st.button("生成手动预测"):
                    manual_data = []
                    for pos in available_positions:
                        if pos in custom_selection:
                            nums = [all_preds[pos][m] for m in custom_selection[pos]]
                            # 手动模式也检查重复
                            if nums[0] == nums[1]:
                                st.error(f"{pos} 两个杀号相同({nums[0]})，预测无效！")
                                continue
                            
                            status = "待验证"
                            if actual_result is not None and pos in actual_result:
                                actual_num = int(actual_result[pos])
                                status = "✅正确" if actual_num not in nums else "❌错误"
                            
                            manual_data.append({
                                '位置': pos,
                                '杀号1': nums[0],
                                '方法1': custom_selection[pos][0],
                                '杀号2': nums[1],
                                '方法2': custom_selection[pos][1],
                                '实际开奖': int(actual_result[pos]) if actual_result is not None else '-',
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
    
    # 提供模板
    template = pd.DataFrame(columns=['开奖期号', '万位', '千位', '百位', '十位'])
    csv = template.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="template.csv">下载数据模板</a>'
    st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ 免责声明：本系统仅供娱乐和数学研究，不构成投注建议。历史数据不具备预测未来能力。")