import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import io

st.set_page_config(page_title="排列五分拆跟单系统", page_icon="🎯", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('排列五开奖历史.xlsx', skiprows=1)
    df.columns = ['期号', '万位', '千位', '百位', '十位']
    for col in ['期号', '万位', '千位', '百位', '十位']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().drop_duplicates(subset=['期号'], keep='last')
    df = df.sort_values('期号', ascending=True).reset_index(drop=True)
    return df

# ========== 5个专家的算法 ==========

def expert_hot(numbers):
    """热号猎手：杀掉最冷的2个"""
    freq = Counter(numbers)
    cold = sorted(freq.items(), key=lambda x: x[1])[:2]
    return [c[0] for c in cold]

def expert_cold(numbers):
    """冷号刺客：杀掉最热的2个"""
    freq = Counter(numbers)
    hot = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:2]
    return [h[0] for h in hot]

def expert_trend(numbers):
    """趋势先知：杀掉逆趋势的2个"""
    if len(numbers) < 2:
        return [numbers[-1], (numbers[-1]+1)%10]
    diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
    common_diff = Counter(diffs).most_common(1)[0][0] if diffs else 0
    last = numbers[-1]
    predicted = (last + common_diff) % 10
    distances = {i: abs((i - predicted) % 10) for i in range(10)}
    kill = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:2]
    return [k[0] for k in kill]

def expert_chaos(numbers):
    """混沌行者：杀掉近期刚出的"""
    recent = numbers[-5:] if len(numbers) >= 5 else numbers
    freq = Counter(recent)
    hot_recent = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:2]
    kill = [h[0] for h in hot_recent]
    if len(kill) < 2:
        kill = numbers[-2:] if len(numbers) >= 2 else [0, 1]
    return kill[:2]

def expert_ml_simple(numbers):
    """ML大师简化版：用简单的线性趋势"""
    if len(numbers) < 5:
        return expert_chaos(numbers)
    x = np.array(range(len(numbers)))
    y = np.array(numbers)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = int(intercept + slope * len(numbers)) % 10
    distances = {i: abs(i - predicted) for i in range(10)}
    kill = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:2]
    return [k[0] for k in kill]

EXPERTS = {
    "热号猎手": expert_hot,
    "冷号刺客": expert_cold,
    "趋势先知": expert_trend,
    "混沌行者": expert_chaos,
    "ML大师": expert_ml_simple
}

# ========== 位置级评估 ==========

def evaluate_expert_position(df, expert_func, position, start_period, end_period, window_size=30):
    results = []

    mask = (df['期号'] >= start_period) & (df['期号'] <= end_period)
    test_periods = df[mask]['期号'].tolist()

    for period in test_periods:
        period_mask = df['期号'] == period
        if not period_mask.any():
            continue

        target_idx = df[period_mask].index[0]
        if target_idx < window_size:
            continue

        window = df.iloc[target_idx - window_size:target_idx]
        numbers = window[position].astype(int).tolist()

        kill = expert_func(numbers)

        actual = int(df[df['期号'] == period][position].values[0])

        is_correct = actual in kill
        results.append({
            'period': int(period),
            'actual': actual,
            'kill': kill,
            'is_correct': is_correct
        })

    if len(results) == 0:
        return 0, 0, 0, []

    correct_count = sum([r['is_correct'] for r in results])
    total = len(results)
    accuracy = correct_count / total * 100

    recent_5 = results[-5:] if len(results) >= 5 else results
    recent_correct = sum([r['is_correct'] for r in recent_5])
    recent_rate = (recent_correct / len(recent_5) * 100) if recent_5 else 0

    max_streak = 0
    current = 0
    for r in results:
        if r['is_correct']:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return accuracy, recent_rate, max_streak, results

# ========== 界面 ==========
st.title("排列五分拆跟单系统")
st.markdown("每个位置独立选专家，万位跟张三，千位跟李四！")

try:
    df = load_data()
    st.info(f"已加载历史数据：{len(df)}期（{int(df['期号'].min())} - {int(df['期号'].max())}）")
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["位置级排行榜", "分拆跟单", "设置"])

# ========== 标签页1：位置级排行榜 ==========
with tab1:
    st.subheader("各位置专家排行榜（独立评估）")

    col1, col2, col3 = st.columns(3)
    with col1:
        eval_start = st.number_input("评估起始期号", 
                                    value=int(df['期号'].max()) - 29,
                                    min_value=int(df['期号'].min()) + 30)
    with col2:
        eval_end = st.number_input("评估结束期号", 
                                  value=int(df['期号'].max()),
                                  max_value=int(df['期号'].max()))
    with col3:
        eval_window = st.selectbox("观察窗口", options=[20, 30, 50], index=1, key="eval_window")

    if st.button("生成分位置排行榜", type="primary", use_container_width=True):
        positions = ['万位', '千位', '百位', '十位']

        position_leaderboards = {}

        progress_bar = st.progress(0)
        total_calculations = len(positions) * len(EXPERTS)
        current = 0

        for pos_idx, position in enumerate(positions):
            position_leaderboards[position] = []

            for expert_name, expert_func in EXPERTS.items():
                current += 1
                progress_bar.progress(current / total_calculations)

                acc, recent_rate, max_streak, details = evaluate_expert_position(
                    df, expert_func, position, eval_start, eval_end, eval_window
                )

                position_leaderboards[position].append({
                    'expert': expert_name,
                    'accuracy': acc,
                    'recent_rate': recent_rate,
                    'max_streak': max_streak,
                    'details': details
                })

            position_leaderboards[position].sort(
                key=lambda x: (x['accuracy'], x['recent_rate']), 
                reverse=True
            )

        progress_bar.empty()

        st.session_state.position_leaderboards = position_leaderboards
        st.session_state.eval_period = (eval_start, eval_end)

        st.markdown("---")

        cols = st.columns(4)
        for idx, position in enumerate(positions):
            with cols[idx]:
                st.markdown(f"### {position}")

                leaderboard = position_leaderboards[position]

                for rank, expert in enumerate(leaderboard[:3], 1):
                    medal = ["1️⃣", "2️⃣", "3️⃣"][rank-1]

                    if expert['accuracy'] >= 50:
                        color = "#d4edda"
                    elif expert['accuracy'] >= 40:
                        color = "#fff3cd"
                    else:
                        color = "#f8d7da"

                    st.markdown(f"""
                    <div style="background:{color};padding:10px;border-radius:5px;margin:5px 0;font-size:14px;">
                        <b>{medal} {expert['expert']}</b><br>
                        总胜率: {expert['accuracy']:.1f}%<br>
                        近期5期: {expert['recent_rate']:.0f}%<br>
                        最大连对: {expert['max_streak']}期
                    </div>
                    """, unsafe_allow_html=True)

                champion = leaderboard[0]
                if champion['accuracy'] > 0:
                    st.success(f"推荐：{champion['expert']}")

# ========== 标签页2：分拆跟单 ==========
with tab2:
    st.subheader("分拆跟单（每个位置独立选择）")

    if 'position_leaderboards' not in st.session_state:
        st.warning("请先到'位置级排行榜'进行评估！")
    else:
        position_leaderboards = st.session_state.position_leaderboards

        col1, col2 = st.columns(2)
        with col1:
            target_period = st.number_input(
                "目标期号",
                min_value=int(df['期号'].min()) + 30,
                max_value=int(df['期号'].max()) + 10,
                value=int(df['期号'].max()) + 1,
            )
        with col2:
            follow_window = st.selectbox("观察窗口", options=[20, 30, 50], index=1, key="follow_window")

        st.markdown("### 选择跟单模式")

        mode = st.radio(
            "跟单模式",
            options=[
                "全自动分配（每个位置跟该位置的冠军）",
                "纯随机分配（每个位置随机选专家）",
                "均衡分配（确保每个专家至少用一次）",
                "手动分配（我为每个位置指定专家）"
            ],
            index=0
        )

        allocation = {}
        positions = ['万位', '千位', '百位', '十位']

        if mode == "全自动分配（每个位置跟该位置的冠军）":
            for position in positions:
                champion = position_leaderboards[position][0]
                allocation[position] = champion['expert']

            st.markdown("#### 自动分配方案")
            cols = st.columns(4)
            for idx, position in enumerate(positions):
                with cols[idx]:
                    expert_name = allocation[position]
                    acc = position_leaderboards[position][0]['accuracy']
                    st.markdown(f"""
                    <div style="background:#e3f2fd;padding:15px;border-radius:10px;text-align:center;">
                        <div style="font-size:18px;font-weight:bold;">{position}</div>
                        <div style="font-size:20px;margin:10px 0;">{expert_name}</div>
                        <div style="font-size:14px;color:#666;">胜率 {acc:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

        elif mode == "纯随机分配（每个位置随机选专家）":
            import random
            for position in positions:
                allocation[position] = random.choice(list(EXPERTS.keys()))

            st.markdown("#### 随机分配方案")
            cols = st.columns(4)
            for idx, position in enumerate(positions):
                with cols[idx]:
                    expert_name = allocation[position]
                    st.info(f"{position}: {expert_name}")

        elif mode == "均衡分配（确保每个专家至少用一次）":
            all_experts = list(EXPERTS.keys())
            for idx, position in enumerate(positions):
                allocation[position] = all_experts[idx % len(all_experts)]

            st.markdown("#### 均衡分配方案")
            cols = st.columns(4)
            for idx, position in enumerate(positions):
                with cols[idx]:
                    expert_name = allocation[position]
                    st.info(f"{position}: {expert_name}")

        else:  # 手动分配
            st.markdown("#### 手动分配")
            for position in positions:
                expert_options = [e['expert'] for e in position_leaderboards[position]]
                default_idx = 0

                allocation[position] = st.selectbox(
                    f"{position}选择专家",
                    options=expert_options,
                    index=default_idx,
                    key=f"manual_{position}"
                )

        if st.button("生成分拆跟单预测", type="primary", use_container_width=True):
            target_mask = df['期号'] == target_period
            if not target_mask.any():
                target_idx = len(df)
            else:
                target_idx = df[target_mask].index[0]

            if target_idx < follow_window:
                st.error(f"数据不足！需要{follow_window}期历史数据")
            else:
                window = df.iloc[target_idx - follow_window:target_idx]

                result = {}
                expert_results = {}

                for position in positions:
                    numbers = window[position].astype(int).tolist()
                    expert_name = allocation[position]
                    expert_func = EXPERTS[expert_name]

                    kill = expert_func(numbers)
                    result[position] = kill

                    if expert_name in expert_results:
                        expert_results[expert_name].append(position)
                    else:
                        expert_results[expert_name] = [position]

                st.markdown("---")
                st.success(f"第 {target_period} 期分拆跟单预测完成")

                st.markdown("### 专家分配与杀号结果")

                cols = st.columns(4)
                for idx, position in enumerate(positions):
                    with cols[idx]:
                        kills = result[position]
                        expert_name = allocation[position]

                        st.markdown(f"""
                        <div style="background:white;padding:20px;border-radius:10px;text-align:center;box-shadow:0 4px 6px rgba(0,0,0,0.1);margin:10px 0;border:2px solid #3498db;">
                            <div style="color:#e74c3c;font-size:16px;font-weight:bold;margin-bottom:5px;">{position}</div>
                            <div style="color:#666;font-size:12px;margin-bottom:10px;">{expert_name}</div>
                            <div>
                                <span style="display:inline-block;width:50px;height:50px;line-height:50px;background:#e74c3c;color:white;border-radius:50%;font-size:24px;font-weight:bold;margin:0 5px;">{kills[0]}</span>
                                <span style="display:inline-block;width:50px;height:50px;line-height:50px;background:#e74c3c;color:white;border-radius:50%;font-size:24px;font-weight:bold;margin:0 5px;">{kills[1]}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("### 本次预测统计")
                for expert_name, pos_list in expert_results.items():
                    st.write(f"- **{expert_name}**：负责 {', '.join(pos_list)}（共{len(pos_list)}个位置）")

# ========== 标签页3：设置 ==========
with tab3:
    st.subheader("系统设置")

    st.markdown("### 专家说明")
    st.write("""
    **5大专家算法：**
    - 热号猎手：相信强者恒强，杀掉最冷的2个号码
    - 冷号刺客：相信均值回归，杀掉最热的2个号码  
    - 趋势先知：识别连号/等差趋势，杀掉逆趋势的号码
    - 混沌行者：完全随机，杀掉最近刚出的号码
    - ML大师：简单线性回归预测，杀掉距离预测值最远的
    """)

    st.markdown("### 使用建议")
    st.write("""
    **分拆跟单的优势：**
    1. 精细化：万位可能处于"热号阶段"，千位可能处于"趋势阶段"
    2. 风险分散：不把鸡蛋放一个篮子，4个位置用4种策略
    3. 动态适配：每个位置都能找到当前最适合的专家
    """)

st.markdown("---")
st.caption("分拆跟单：细粒度策略轮动，每个位置都配专属专家！")
