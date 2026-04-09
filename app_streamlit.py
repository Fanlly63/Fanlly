import streamlit as st
import pandas as pd
import numpy as np
from algorithms import KillAlgorithms
from rotation_manager import RotationManager
import base64

st.set_page_config(page_title="排列五智能杀号系统", layout="wide")

# 初始化session state
if 'history' not in st.session_state:
    st.session_state.history = None

st.title("🔢 排列五智能杀号预测系统")
st.markdown("基于8种数学算法的轮换杀号策略 | 相邻期数方法不重复")

# 侧边栏
with st.sidebar:
    st.header("📊 数据配置")
    uploaded_file = st.file_uploader("上传历史数据(Excel)", type=['xlsx', 'xls'])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.history = df
            st.success(f"已加载 {len(df)} 期数据")
            
            # 显示最新期号
            if '开奖期号' in df.columns:
                latest_period = df.iloc[-1]['开奖期号']
                st.info(f"最新期号: {latest_period}")
            else:
                st.info(f"共 {len(df)} 期数据")
        except Exception as e:
            st.error(f"读取文件失败: {str(e)}")
    
    st.markdown("---")
    st.header("⚙️ 预测设置")
    prediction_mode = st.radio(
        "预测模式",
        ["自动轮换（防重复）", "手动选择方法"]
    )

# 主界面
if st.session_state.history is not None:
    df = st.session_state.history
    
    try:
        # 初始化
        algo = KillAlgorithms(df)
        rot_manager = RotationManager()
        
        # 获取所有预测
        all_preds = algo.get_all_predictions()
        available_positions = list(all_preds.keys())  # 动态获取实际有的位置
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 下期杀号预测")
            
            if prediction_mode == "自动轮换（防重复）":
                current_selection = {}
                final_kill_numbers = {}
                
                # 只为实际存在的位置生成预测
                for pos in available_positions:
                    try:
                        methods = rot_manager.select_methods_for_position(pos)
                        current_selection[pos] = methods
                        
                        # 获取杀号结果
                        kill_nums = []
                        for m in methods:
                            if m in all_preds[pos]:
                                kill_nums.append(all_preds[pos][m])
                            else:
                                st.error(f"方法 {m} 未找到")
                                kill_nums.append(0)
                        
                        final_kill_numbers[pos] = kill_nums
                    except Exception as e:
                        st.error(f"处理 {pos} 时出错: {str(e)}")
                        continue
                
                # 显示结果
                if current_selection:
                    result_data = []
                    for pos in available_positions:
                        if pos in final_kill_numbers:
                            nums = final_kill_numbers[pos]
                            safe_nums = [i for i in range(10) if i not in nums]
                            result_data.append({
                                '位置': pos,
                                '杀号1': nums[0] if len(nums) > 0 else '-',
                                '方法1': current_selection[pos][0] if pos in current_selection else '-',
                                '杀号2': nums[1] if len(nums) > 1 else '-',
                                '方法2': current_selection[pos][1] if pos in current_selection else '-',
                                '保留号': safe_nums
                            })
                    
                    if result_data:
                        result_df = pd.DataFrame(result_data)
                        st.table(result_df)
                        
                        # 保存按钮
                        if st.button("✅ 确认本期预测并保存轮换状态", type="primary"):
                            rot_manager.save_state(current_selection)
                            st.success("轮换状态已保存！下期将自动避开本期使用的方法")
            
            else:  # 手动模式
                st.warning("手动模式：请为每个位置选择2种不同的方法")
                
                custom_selection = {}
                for pos in available_positions:
                    with st.expander(f"{pos} 设置"):
                        m1 = st.selectbox(f"{pos} - 杀号1方法", rot_manager.methods, key=f"{pos}_1")
                        remaining = [m for m in rot_manager.methods if m != m1]
                        m2 = st.selectbox(f"{pos} - 杀号2方法", remaining, key=f"{pos}_2")
                        custom_selection[pos] = [m1, m2]
                
                if st.button("生成手动预测"):
                    manual_data = []
                    for pos in available_positions:
                        if pos in custom_selection:
                            nums = [all_preds[pos][m] for m in custom_selection[pos] if m in all_preds[pos]]
                            manual_data.append({
                                '位置': pos,
                                '杀号1': nums[0] if len(nums) > 0 else '-',
                                '方法1': custom_selection[pos][0],
                                '杀号2': nums[1] if len(nums) > 1 else '-',
                                '方法2': custom_selection[pos][1]
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
            if 'last_used' in st.session_state and st.session_state.last_used:
                for pos, methods in st.session_state.last_used.items():
                    st.text(f"{pos}: {', '.join(methods)}")
            else:
                st.info("暂无历史记录")
                
    except Exception as e:
        st.error(f"系统运行错误: {str(e)}")
        st.error("请检查Excel文件格式，确保包含'万位','千位','百位','十位'等列名")
        
else:
    st.info("👈 请先上传历史数据文件")
    
    # 提供模板
    template = pd.DataFrame(columns=['开奖期号', '万位', '千位', '百位', '十位', '个位'])
    csv = template.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="template.csv">下载数据模板</a>'
    st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ 免责声明：本系统仅供娱乐和数学研究，不构成投注建议。")