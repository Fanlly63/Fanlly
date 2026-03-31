import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import random

st.set_page_config(page_title="排列五杀号追踪器", layout="wide")

# 232期真实数据
HISTORICAL_DATA = [
    [2025200,2,1,1,2], [2025201,4,0,1,3], [2025202,2,5,3,7], [2025203,7,3,5,1], [2025204,8,3,0,7],
    [2025205,7,8,3,2], [2025206,6,0,5,3], [2025207,1,5,4,5], [2025208,3,0,9,4], [2025209,9,2,1,4],
    [2025210,6,2,2,2], [2025211,0,5,0,9], [2025212,2,2,3,5], [2025213,8,6,8,5], [2025214,1,7,6,6],
    [2025215,6,9,5,9], [2025216,4,1,0,6], [2025217,2,4,2,0], [2025218,3,0,7,8], [2025219,7,8,4,9],
    [2025220,3,5,3,2], [2025221,7,9,1,7], [2025222,6,6,0,8], [2025223,7,9,5,0], [2025224,0,2,1,9],
    [2025225,0,0,9,0], [2025226,3,7,3,2], [2025227,4,5,3,9], [2025228,1,6,4,6], [2025229,1,2,3,0],
    [2025230,1,7,8,0], [2025231,2,5,8,5], [2025232,3,9,7,6], [2025233,1,4,3,9], [2025234,2,8,2,5],
    [2025235,3,2,1,4], [2025236,1,5,1,1], [2025237,7,1,3,3], [2025238,7,5,4,5], [2025239,2,2,0,1],
    [2025240,6,2,7,6], [2025241,2,8,6,3], [2025242,5,2,3,9], [2025243,8,1,2,8], [2025244,7,5,8,6],
    [2025245,9,9,4,9], [2025246,2,0,7,6], [2025247,3,6,4,2], [2025248,9,8,3,8], [2025249,5,3,2,0],
    [2025250,1,7,8,0], [2025251,5,5,8,5], [2025252,1,9,0,5], [2025253,6,9,3,0], [2025254,6,2,8,8],
    [2025255,5,8,3,2], [2025256,7,9,7,7], [2025257,5,6,1,5], [2025258,0,3,8,6], [2025259,4,2,0,9],
    [2025260,5,1,4,5], [2025261,8,7,7,5], [2025262,5,1,2,6], [2025263,1,4,8,0], [2025264,2,0,2,0],
    [2025265,2,8,1,4], [2025266,7,4,2,5], [2025267,8,5,6,1], [2025268,3,2,1,3], [2025269,8,7,5,0],
    [2025270,0,1,8,2], [2025271,8,9,8,5], [2025272,7,6,7,4], [2025273,4,1,7,3], [2025274,7,8,7,7],
    [2025275,1,5,0,7], [2025276,9,8,2,6], [2025277,4,3,2,2], [2025278,6,2,7,8], [2025279,3,7,5,7],
    [2025280,5,3,9,3], [2025281,7,0,7,9], [2025282,0,9,6,9], [2025283,0,5,7,8], [2025284,2,4,7,3],
    [2025285,9,0,5,3], [2025286,6,4,1,0], [2025287,1,8,0,3], [2025288,5,0,4,2], [2025289,6,8,5,6],
    [2025290,1,2,5,1], [2025291,9,4,8,5], [2025292,6,7,3,5], [2025293,0,9,4,4], [2025294,9,9,4,4],
    [2025295,1,2,4,5], [2025296,0,5,2,3], [2025297,4,5,3,8], [2025298,1,4,1,2], [2025299,3,0,0,9],
    [2025300,8,4,3,0], [2025301,5,8,2,0], [2025302,0,1,6,3], [2025303,9,4,3,8], [2025304,0,7,9,1],
    [2025305,4,9,7,4], [2025306,1,7,8,6], [2025307,6,1,8,1], [2025308,2,3,4,6], [2025309,5,2,7,2],
    [2025310,0,1,6,5], [2025311,1,1,4,5], [2025312,1,0,9,5], [2025313,4,4,5,5], [2025314,8,0,2,4],
    [2025315,9,7,3,6], [2025316,9,1,8,1], [2025317,7,5,6,6], [2025318,4,3,8,4], [2025319,8,6,5,5],
    [2025320,7,2,5,6], [2025321,5,6,8,0], [2025322,1,4,8,3], [2025323,6,8,7,1], [2025324,5,1,3,9],
    [2025325,8,3,6,0], [2025326,0,4,5,1], [2025327,8,2,9,2], [2025328,8,2,5,1], [2025329,3,7,0,9],
    [2025330,5,9,4,5], [2025331,9,8,0,9], [2025332,9,3,9,4], [2025333,9,6,1,0], [2025334,5,8,3,3],
    [2025335,2,7,6,4], [2025336,4,1,2,6], [2025337,5,5,4,3], [2025338,1,6,4,6], [2025339,9,0,0,4],
    [2025340,2,5,5,6], [2025341,8,1,7,5], [2025342,0,7,9,2], [2025343,0,1,2,5], [2025344,4,9,5,7],
    [2025345,4,3,1,9], [2025346,8,2,6,0], [2025347,6,9,5,6], [2025348,3,7,3,1], [2025349,0,3,2,2],
    [2025350,6,7,3,4], [2025351,6,1,7,2], [2026001,8,3,0,1], [2026002,6,8,9,2], [2026003,0,8,6,8],
    [2026004,8,7,8,8], [2026005,0,6,1,6], [2026006,7,1,8,8], [2026007,1,2,6,6], [2026008,5,0,9,6],
    [2026009,2,4,8,2], [2026010,9,2,0,3], [2026011,0,3,0,6], [2026012,5,2,2,0], [2026013,4,7,7,8],
    [2026014,8,3,3,8], [2026015,2,0,2,4], [2026016,2,1,5,9], [2026017,1,1,2,8], [2026018,8,9,6,3],
    [2026019,8,4,8,9], [2026020,0,9,5,8], [2026021,2,6,3,0], [2026022,2,5,5,0], [2026023,4,9,5,2],
    [2026024,2,6,8,3], [2026025,1,8,1,1], [2026026,2,9,1,4], [2026027,0,8,4,0], [2026028,4,9,9,9],
    [2026029,6,1,3,7], [2026030,8,1,1,5], [2026031,7,8,6,9], [2026032,7,3,0,7], [2026033,1,4,1,3],
    [2026034,2,1,6,9], [2026035,7,1,7,0], [2026036,7,8,8,2], [2026037,7,2,4,5], [2026038,7,8,9,8],
    [2026039,8,5,2,8], [2026040,0,0,9,7], [2026041,6,3,9,6], [2026042,7,9,5,2], [2026043,1,7,9,2],
    [2026044,0,5,1,6], [2026045,1,4,4,7], [2026046,7,0,3,2], [2026047,3,3,0,4], [2026048,0,9,7,6],
    [2026049,5,9,2,5], [2026050,4,3,5,4], [2026051,0,4,5,3], [2026052,6,4,9,1], [2026053,7,4,4,9],
    [2026054,6,5,5,6], [2026055,5,3,7,5], [2026056,1,1,9,2], [2026057,4,7,9,3], [2026058,5,9,0,6],
    [2026059,2,0,9,9], [2026060,1,1,8,0], [2026061,3,4,7,4], [2026062,3,1,5,4], [2026063,7,2,3,8],
    [2026064,5,2,8,9], [2026065,8,5,1,8], [2026066,8,0,7,8], [2026067,4,2,8,1], [2026068,8,4,0,0],
    [2026069,1,1,4,6], [2026070,1,3,1,8], [2026071,5,5,1,5], [2026072,1,1,5,4], [2026073,3,2,5,0],
    [2026074,4,0,6,3], [2026075,4,3,7,4], [2026076,4,1,9,6], [2026077,7,8,0,7], [2026078,6,3,4,8],
    [2026079,1,9,5,9], [2026080,7,7,0,0]
]

df_hist = pd.DataFrame(HISTORICAL_DATA, columns=['期号', '万位', '千位', '百位', '十位'])

# 初始化session state
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 20  # 从第20期开始（前面留作历史数据）
    st.session_state.experts_data = {}
    st.session_state.history = {}  # 记录每期预测和结果
    
    # 初始化10个专家
    for i in range(10):
        st.session_state.experts_data[i] = {
            'name': f'砖家{i+1}号',
            'strategy': random.choice(['热号追冷', '冷号追热', '随机', '遗漏值']),
            'kill': [2,2,2,2] if i < 5 else [3,3,3,3],  # 前5个杀2个，后5个杀3个
            'streak': 0,  # 正数连对，负数连错
            'total_correct': 0,
            'total_test': 0
        }

def get_prediction(expert_id, history_data):
    """生成预测"""
    exp = st.session_state.experts_data[expert_id]
    kills = exp['kill']
    
    pred = []
    for pos_idx in range(4):
        # 基于历史生成（这里简化用随机，实际可替换为策略）
        recent = [h[pos_idx+1] for h in history_data[-10:]]
        counter = Counter(recent)
        
        # 简单策略：杀出现最多的k个（追冷/赌徒谬误）
        most_common = [n for n, _ in counter.most_common()]
        killed = most_common[:kills[pos_idx]]
        
        # 如果不够就随机补
        if len(killed) < kills[pos_idx]:
            remaining = [x for x in range(10) if x not in killed]
            killed += random.sample(remaining, kills[pos_idx] - len(killed))
        
        pred.append(sorted(killed))
    
    return pred

def check_result(pred, actual):
    """检查四全对"""
    correct_pos = [actual[i] not in pred[i] for i in range(4)]
    all_correct = all(correct_pos)
    return all_correct, correct_pos

# 侧边栏控制
st.sidebar.header("🎮 控制面板")

# 期号选择
available_periods = df_hist['期号'].tolist()[20:]  # 从第20期开始可用
selected_period = st.sidebar.selectbox(
    "选择验证期号",
    available_periods,
    index=st.session_state.current_idx - 20 if st.session_state.current_idx < len(df_hist) else 0
)

# 找到对应的索引
target_idx = df_hist[df_hist['期号'] == selected_period].index[0]

st.sidebar.markdown("---")

# 运行控制
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("▶️ 运行本期", use_container_width=True):
        if st.session_state.current_idx <= target_idx:
            # 运行到选中的期号
            while st.session_state.current_idx <= target_idx:
                hist_data = df_hist.iloc[:st.session_state.current_idx].values.tolist()
                actual = df_hist.iloc[st.session_state.current_idx][['万位', '千位', '百位', '十位']].values
                
                period_no = df_hist.iloc[st.session_state.current_idx]['期号']
                st.session_state.history[period_no] = {}
                
                for eid in range(10):
                    pred = get_prediction(eid, hist_data)
                    is_correct, pos_detail = check_result(pred, actual)
                    
                    exp = st.session_state.experts_data[eid]
                    exp['total_test'] += 1
                    
                    if is_correct:
                        exp['total_correct'] += 1
                        exp['streak'] = exp['streak'] + 1 if exp['streak'] > 0 else 1
                    else:
                        exp['streak'] = exp['streak'] - 1 if exp['streak'] < 0 else -1
                    
                    st.session_state.history[period_no][eid] = {
                        'prediction': pred,
                        'actual': actual,
                        'is_correct': is_correct,
                        'pos_detail': pos_detail
                    }
                
                st.session_state.current_idx += 1
            st.rerun()

with c2:
    if st.button("⏭️ 连续10期", use_container_width=True):
        for _ in range(10):
            if st.session_state.current_idx < len(df_hist):
                hist_data = df_hist.iloc[:st.session_state.current_idx].values.tolist()
                actual = df_hist.iloc[st.session_state.current_idx][['万位', '千位', '百位', '十位']].values
                period_no = df_hist.iloc[st.session_state.current_idx]['期号']
                st.session_state.history[period_no] = {}
                
                for eid in range(10):
                    pred = get_prediction(eid, hist_data)
                    is_correct, pos_detail = check_result(pred, actual)
                    exp = st.session_state.experts_data[eid]
                    exp['total_test'] += 1
                    if is_correct:
                        exp['total_correct'] += 1
                        exp['streak'] = exp['streak'] + 1 if exp['streak'] > 0 else 1
                    else:
                        exp['streak'] = exp['streak'] - 1 if exp['streak'] < 0 else -1
                    st.session_state.history[period_no][eid] = {
                        'prediction': pred,
                        'is_correct': is_correct,
                        'pos_detail': pos_detail
                    }
                st.session_state.current_idx += 1
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**说明**")
st.sidebar.markdown("- 四全对=万千百十位全部杀对")
st.sidebar.markdown("- 连对/连错连续统计")
st.sidebar.markdown("- 当前期数：" + str(st.session_state.current_idx))

# 主界面
st.title("排列五 · 十砖家杀号追踪器")
st.caption("验证：连错7期的专家下期是否该中了？")

# 显示当前选中期的结果
if selected_period in st.session_state.history:
    result_data = st.session_state.history[selected_period]
    actual_result = df_hist[df_hist['期号'] == selected_period][['万位', '千位', '百位', '十位']].values[0]
    
    # 开奖结果展示
    st.markdown(f"### 第 {selected_period} 期开奖结果")
    cols = st.columns(4)
    pos_names = ['万位', '千位', '百位', '十位']
    for i, (col, val, name) in enumerate(zip(cols, actual_result, pos_names)):
        col.metric(name, str(val))
    
    # 专家表现
    st.markdown("### 各专家杀号表现")
    
    # 表头
    header_cols = st.columns([1.5, 2, 3, 1.5, 2])
    header_cols[0].markdown("**专家**")
    header_cols[1].markdown("**杀号策略**")
    header_cols[2].markdown("**本期杀号**")
    header_cols[3].markdown("**结果**")
    header_cols[4].markdown("**连对/连错**")
    
    st.markdown("---")
    
    # 按连错期数排序（连错多的在前面）
    expert_list = []
    for eid in range(10):
        exp = st.session_state.experts_data[eid]
        res = result_data[eid]
        expert_list.append((eid, exp, res))
    
    # 排序：连错的排在前面（负数越小越靠前）
    expert_list.sort(key=lambda x: x[1]['streak'])
    
    for eid, exp, res in expert_list:
        row_cols = st.columns([1.5, 2, 3, 1.5, 2])
        
        # 专家名称
        row_cols[0].markdown(f"**{exp['name']}**")
        
        # 策略
        kill_type = "杀2个" if exp['kill'][0] == 2 else "杀3个"
        row_cols[1].markdown(f"{exp['strategy']}<br><small>{kill_type}</small>", unsafe_allow_html=True)
        
        # 具体杀号
        pred = res['prediction']
        pred_str = f"万{pred[0]}<br>千{pred[1]}<br>百{pred[2]}<br>十{pred[3]}"
        row_cols[2].markdown(f"<small>{pred_str}</small>", unsafe_allow_html=True)
        
        # 结果
        if res['is_correct']:
            row_cols[3].markdown("✅ **中**")
        else:
            row_cols[3].markdown("❌ 错")
        
        # 连对/连错
        streak = exp['streak']
        if streak > 0:
            row_cols[4].markdown(f"🟢 连对{streak}期")
        elif streak < 0:
            row_cols[4].markdown(f"🔴 连错{abs(streak)}期")
        else:
            row_cols[4].markdown("⚪ 平")
    
    # 连错排行榜（推荐区）
    st.markdown("---")
    st.markdown("### 🔥 当前连错排行榜（均值回归推荐）")
    
    losers = [(eid, exp) for eid, exp in st.session_state.experts_data.items() if exp['streak'] < 0]
    losers.sort(key=lambda x: x[1]['streak'])  # 连错多的在前
    
    if losers:
        rec_cols = st.columns(min(3, len(losers)))
        for i, (eid, exp) in enumerate(losers[:3]):
            with rec_cols[i]:
                streak = abs(exp['streak'])
                color = "red" if streak >= 5 else "orange" if streak >= 3 else "gray"
                st.markdown(f"""
                <div style="padding:10px; border-left:5px solid {color}; background-color:#f0f0f0;">
                    <h4>{exp['name']}</h4>
                    <p style="font-size:24px; color:{color}; font-weight:bold; margin:0;">
                        连错 {streak} 期
                    </p>
                    <small>累计胜率: {exp['total_correct']/exp['total_test']*100:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
                
                if streak >= 7:
                    st.error("⚠️ 已达7期连错！按理论下期该中了？")
    else:
        st.info("当前没有连错的专家")

else:
    st.info("👈 请在左侧选择期号并点击'运行本期'")

# 底部统计
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📊 累计统计")
    
    stats_cols = st.columns(5)
    for i in range(10):
        exp = st.session_state.experts_data[i]
        if exp['total_test'] > 0:
            rate = exp['total_correct'] / exp['total_test'] * 100
            with stats_cols[i % 5]:
                st.metric(
                    f"{exp['name']}", 
                    f"{rate:.1f}%",
                    f"连错{abs(exp['streak'])}期" if exp['streak'] < 0 else (f"连对{exp['streak']}期" if exp['streak'] > 0 else "平")
                )