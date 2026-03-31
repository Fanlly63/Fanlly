import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import random

st.set_page_config(page_title="排列五杀号对照表", layout="wide")

# 真实开奖数据（完整232期）
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

# 10个固定策略的专家
EXPERTS_CONFIG = [
    {"name": "砖家1号[热杀2]", "strategy": "hot", "kill": [2,2,2,2]},
    {"name": "砖家2号[热杀3]", "strategy": "hot", "kill": [3,3,3,3]},
    {"name": "砖家3号[冷杀2]", "strategy": "cold", "kill": [2,2,2,2]},
    {"name": "砖家4号[冷杀3]", "strategy": "cold", "kill": [3,3,3,3]},
    {"name": "砖家5号[随机2]", "strategy": "random", "kill": [2,2,2,2]},
    {"name": "砖家6号[随机3]", "strategy": "random", "kill": [3,3,3,3]},
    {"name": "砖家7号[追热2]", "strategy": "trend", "kill": [2,2,2,2]},
    {"name": "砖家8号[追热3]", "strategy": "trend", "kill": [3,3,3,3]},
    {"name": "砖家9号[遗漏2]", "strategy": "missing", "kill": [2,2,2,2]},
    {"name": "砖家10号[遗漏3]", "strategy": "missing", "kill": [3,3,3,3]}
]

def generate_prediction(config, history_data):
    """生成杀号预测"""
    strategy = config["strategy"]
    kills = config["kill"]
    
    pred = []
    for pos_idx in range(4):
        if len(history_data) < 5:
            killed = sorted(random.sample(range(10), kills[pos_idx]))
        else:
            recent = [row[pos_idx+1] for row in history_data[-20:]]
            counter = Counter(recent)
            
            if strategy == "hot":
                rare = [n for n, _ in counter.most_common()[::-1]]
                killed = sorted(rare[:kills[pos_idx]])
            elif strategy == "cold":
                common = [n for n, _ in counter.most_common()]
                killed = sorted(common[:kills[pos_idx]])
            elif strategy == "random":
                killed = sorted(random.sample(range(10), kills[pos_idx]))
            elif strategy == "trend":
                last = history_data[-1][pos_idx+1]
                neighbors = list(set([(last-1)%10, last, (last+1)%10]))
                if len(neighbors) < kills[pos_idx]:
                    neighbors += random.sample([x for x in range(10) if x not in neighbors], kills[pos_idx]-len(neighbors))
                killed = sorted(neighbors[:kills[pos_idx]])
            elif strategy == "missing":
                all_nums = set(range(10))
                appeared = set(recent[-5:])
                missing = list(all_nums - appeared)
                if len(missing) >= kills[pos_idx]:
                    killed = sorted(missing[:kills[pos_idx]])
                else:
                    killed = sorted(missing + random.sample(list(appeared), kills[pos_idx]-len(missing)))
        
        if len(killed) < kills[pos_idx]:
            remaining = [x for x in range(10) if x not in killed]
            killed = sorted(killed + random.sample(remaining, kills[pos_idx]-len(killed)))
        
        pred.append(killed)
    
    return pred

def check_all_correct(pred, actual):
    """检查四全对"""
    return all([actual[i] not in pred[i] for i in range(4)])

# 初始化session state
if 'current_period' not in st.session_state:
    st.session_state.current_period = 2025220  # 默认从第20期开始

# 获取可用期号范围
available_periods = df_hist['期号'].tolist()
min_period = min(available_periods)
max_period = max(available_periods)

# 确保当前期号有效
if st.session_state.current_period not in available_periods:
    st.session_state.current_period = available_periods[20]

# 主界面
st.title("🎯 排列五杀号预测对照表")

# 期号选择区域（输入框 + 加减按钮）
st.markdown("### 📅 期号选择")

col1, col2, col3, col4 = st.columns([1, 2, 1, 4])

with col1:
    if st.button("➖ 上期", use_container_width=True):
        current_idx = available_periods.index(st.session_state.current_period)
        if current_idx > 0:
            st.session_state.current_period = available_periods[current_idx - 1]
            st.rerun()

with col2:
    # 数字输入框
    input_period = st.number_input(
        "输入期号",
        min_value=min_period,
        max_value=max_period,
        value=st.session_state.current_period,
        step=1
    )
    # 如果用户手动修改了输入框，更新session state
    if input_period != st.session_state.current_period and input_period in available_periods:
        st.session_state.current_period = int(input_period)
        st.rerun()

with col3:
    if st.button("➕ 下期", use_container_width=True):
        current_idx = available_periods.index(st.session_state.current_period)
        if current_idx < len(available_periods) - 1:
            st.session_state.current_period = available_periods[current_idx + 1]
            st.rerun()

with col4:
    current_idx = available_periods.index(st.session_state.current_period)
    st.info(f"当前第 {current_idx + 1} / {len(available_periods)} 期  (范围：{min_period} - {max_period})")

st.markdown("---")

# 获取当期数据
selected_period = st.session_state.current_period
selected_idx = df_hist[df_hist['期号'] == selected_period].index[0]
actual_result = df_hist.iloc[selected_idx][['万位', '千位', '百位', '十位']].values

# 获取历史数据（该期之前的数据）
hist_before = df_hist.iloc[:selected_idx].values.tolist()

# 生成10个专家的预测
results = []
for eid, config in enumerate(EXPERTS_CONFIG):
    pred = generate_prediction(config, hist_before)
    is_correct = check_all_correct(pred, actual_result)
    
    results.append({
        "专家": config["name"],
        "万位杀号": str(pred[0]).replace("[", "").replace("]", ""),
        "千位杀号": str(pred[1]).replace("[", "").replace("]", ""), 
        "百位杀号": str(pred[2]).replace("[", "").replace("]", ""),
        "十位杀号": str(pred[3]).replace("[", "").replace("]", ""),
        "四全对": "✅ 中" if is_correct else "❌ 错"
    })

# 显示预测表格
df_display = pd.DataFrame(results)
st.dataframe(df_display, use_container_width=True, hide_index=True, height=400)

st.markdown("---")

# 底部显示当期开奖（大字醒目）
st.markdown("### 🎲 当期开奖验证")

cols = st.columns(4)
for i, (col, pos_name, val) in enumerate(zip(cols, ['万位', '千位', '百位', '十位'], actual_result)):
    col.metric(pos_name, int(val))

# 验证说明
st.caption("验证规则：开奖号码不在杀号列表中=该位置杀对，四个位置全杀对=四全对✅")