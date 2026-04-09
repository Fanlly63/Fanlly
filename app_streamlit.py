import streamlit as st
import pandas as pd
import numpy as np
from algorithms import KillAlgorithms
from rotation_manager import RotationManager
import base64

st.set_page_config(page_title="排列五智能杀号系统", layout="wide")

# 初始化
if 'history' not in st.session_state:
    st.session_state.history = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

st.title("🔢 排列五智能杀号预测系统")
st.markdown("基于8种数学算法的轮换杀号策略 | 相邻期数方法不重复")

# 侧边栏 - 数据上传与配置
with st.sidebar:
    st.header("📊 数据配置")
    uploaded_file = st.file_uploader("上传历史数据(Excel)", type=['xlsx', 'xls'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.history = df
        st.success(f"已加载 {len(df)} 期数据")
        
        # 显示最新期号
        latest_period = df.iloc[-1]['开奖期号'] if '开奖期号' in df.columns else f"第{len(df)}期"
        st.info(f"最新期号: {latest_period}")
    
    st.markdown("---")
    st.header("⚙️ 预测设置")
    prediction_mode = st.radio(
        "预测模式",
        ["自动轮换（防重复）", "手动选择方法"]
    )

# 主界面
if st.session_state.history is not None:
    df = st.session_state.history
    
    # 初始化算法和轮换管理器
    algo = KillAlgorithms(df)
    rot_manager = RotationManager()
    
    # 获取所有方法的原始预测（8个方法×5个位置）
    all_preds = algo.get_all_predictions()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 下期杀号预测")
        
        if prediction_mode == "自动轮换（防重复）":
            # 为每个位置选择2种方法（确保不重复）
            current_selection = {}
            final_kill_numbers = {}
            
            for pos in rot_manager.positions:
                methods = rot_manager.select_methods_for_position(pos)
                current_selection[pos] = methods
                
                # 获取这2种方法的杀号结果
                kill_nums = [all_preds[pos][m] for m in methods]
                final_kill_numbers[pos] = kill_nums
            
            # 显示结果表格
            result_data = []
            for pos in rot_manager.positions:
                result_data.append({
                    '位置': pos,
                    '杀号1': final_kill_numbers[pos][0],
                    '使用方法1': current_selection[pos][0],
                    '杀号2': final_kill_numbers[pos][1],
                    '使用方法2': current_selection[pos][1],
                    '建议保留号': [i for i in range(10) if i not in final_kill_numbers[pos]]
                })
            
            result_df = pd.DataFrame(result_data)
            st.table(result_df)
            
            # 保存状态供下期使用
            if st.button("✅ 确认本期预测并保存轮换状态", type="primary"):
                rot_manager.save_state(current_selection)
                st.success("轮换状态已保存！下期将自动避开本期使用的方法")
                
                # 显示下期提示
                st.info("💡 下期预测时，系统将自动避开: " + 
                       ", ".join([f"{pos}:{','.join(m)}" for pos, m in current_selection.items()]))
        
        else:  # 手动模式
            st.warning("手动模式下请自行确保轮换规则，建议相邻期不重复使用相同方法组合")
            
            custom_selection = {}
            for pos in rot_manager.positions:
                with st.expander(f"{pos} 方法选择"):
                    m1 = st.selectbox(f"{pos} - 方法1", rot_manager.methods, key=f"{pos}_1")
                    m2 = st.selectbox(f"{pos} - 方法2", 
                                    [m for m in rot_manager.methods if m != m1], 
                                    key=f"{pos}_2")
                    custom_selection[pos] = [m1, m2]
            
            if st.button("生成手动预测"):
                manual_data = []
                for pos in rot_manager.positions:
                    nums = [all_preds[pos][m] for m in custom_selection[pos]]
                    manual_data.append({
                        '位置': pos,
                        '杀号1': nums[0],
                        '方法1': custom_selection[pos][0],
                        '杀号2': nums[1],
                        '方法2': custom_selection[pos][1]
                    })
                st.table(pd.DataFrame(manual_data))
    
    with col2:
        st.subheader("📈 方法说明")
        methods_info = rot_manager.get_method_pool()
        for name, desc in methods_info.items():
            with st.expander(name):
                st.write(desc)
        
        # 历史准确率统计（模拟）
        st.markdown("---")
        st.subheader("📊 上期回顾")
        if 'last_used' in st.session_state and st.session_state.last_used:
            st.write("上期使用方法:")
            for pos, methods in st.session_state.last_used.items():
                st.text(f"{pos}: {', '.join(methods)}")
        else:
            st.info("暂无历史记录")

else:
    st.info("👈 请先上传历史数据文件（Excel格式，包含开奖期号、万位、千位、百位、十位、个位列）")
    
    # 提供模板下载
    template = pd.DataFrame(columns=['开奖期号', '万位', '千位', '百位', '十位', '个位'])
    csv = template.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="template.csv">下载数据模板</a>'
    st.markdown(href, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.caption("⚠️ 免责声明：本系统仅供娱乐和数学研究，不构成投注建议。彩票开奖为独立随机事件，历史数据不具备预测未来能力。")