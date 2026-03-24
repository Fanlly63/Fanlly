import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

st.set_page_config(page_title="排列五阶段性自适应杀号系统", page_icon="🎯", layout="wide")

@st.cache_data
def load_data():
    """加载数据"""
    df = pd.read_excel('排列五开奖历史.xlsx', skiprows=1)
    df.columns = ['期号', '万位', '千位', '百位', '十位']
    for col in ['期号', '万位', '千位', '百位', '十位']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().drop_duplicates(subset=['期号'], keep='last')
    df = df.sort_values('期号', ascending=True).reset_index(drop=True)
    return df

def analyze_regime(window_data, position):
    """阶段识别器"""
    numbers = window_data[position].astype(int).tolist()
    n = len(numbers)

    freq = Counter(numbers)
    max_freq = max(freq.values())
    min_freq = min(freq.values())
    max_freq_ratio = max_freq / n
    min_freq_ratio = min_freq / n

    diffs = [numbers[i+1] - numbers[i] for i in range(n-1)]

    arithmetic_count = 0
    for i in range(len(diffs)-1):
        if diffs[i] == diffs[i+1] and abs(diffs[i]) <= 3:
            arithmetic_count += 1
    arithmetic_ratio = arithmetic_count / max(1, len(diffs)-1)

    consecutive_count = sum(1 for d in diffs if abs(d) == 1)
    consecutive_ratio = consecutive_count / len(diffs)

    entropy = -sum([(count/n) * np.log2(count/n) for count in freq.values()])
    max_entropy = np.log2(10)
    normalized_entropy = entropy / max_entropy

    regime = "CHAOS"
    confidence = 0.5

    if max_freq_ratio > 0.20 and normalized_entropy < 0.85:
        regime = "HOT"
        confidence = max_freq_ratio
    elif min_freq_ratio < 0.03:
        max_miss = max([i for i in range(10)], key=lambda x: next((j for j, n in enumerate(reversed(numbers)) if n == x), n))
        if max_miss > 15:
            regime = "COLD"
            confidence = 1 - min_freq_ratio
    elif arithmetic_ratio > 0.3 or consecutive_ratio > 0.4:
        regime = "TREND"
        confidence = max(arithmetic_ratio, consecutive_ratio)
    elif normalized_entropy > 0.9 and max_freq_ratio < 0.15:
        regime = "CHAOS"
        confidence = normalized_entropy

    features = {
        'max_freq_ratio': max_freq_ratio,
        'min_freq_ratio': min_freq_ratio,
        'arithmetic_ratio': arithmetic_ratio,
        'consecutive_ratio': consecutive_ratio,
        'entropy': normalized_entropy,
        'hot_number': max(freq.items(), key=lambda x: x[1])[0] if regime == "HOT" else None,
        'cold_number': min(freq.items(), key=lambda x: x[1])[0] if regime == "COLD" else None,
    }

    return regime, confidence, features

def get_kill_strategy(regime, features, position_data):
    """根据阶段选择杀号策略"""
    numbers = position_data.astype(int).tolist()
    freq = Counter(numbers)

    if regime == "HOT":
        cold_numbers = sorted(freq.items(), key=lambda x: x[1])[:2]
        kill = [n[0] for n in cold_numbers]
        strategy_desc = "🔥 热号延续策略：杀掉出现次数最少的冷号"

    elif regime == "COLD":
        hot_numbers = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:2]
        kill = [n[0] for n in hot_numbers]
        strategy_desc = "❄️ 冷号回补策略：杀掉出现次数最多的热号"

    elif regime == "TREND":
        diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        median_diff = int(np.median(diffs))
        last_number = numbers[-1]

        predicted_next = (last_number + median_diff) % 10
        distances = {i: abs(i - predicted_next) for i in range(10)}
        kill = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:2]
        kill = [k[0] for k in kill]
        strategy_desc = f"📈 趋势跟踪策略：趋势公差{median_diff:+d}，杀掉逆趋势号码"

    else:
        recent = numbers[-5:]
        recent_freq = Counter(recent)
        hot_in_recent = sorted(recent_freq.items(), key=lambda x: x[1], reverse=True)[:2]
        kill = [n[0] for n in hot_in_recent]

        if len(kill) < 2:
            kill = numbers[-2:]
        strategy_desc = "🎲 混沌均衡策略：杀掉近期刚出的号码"

    return kill, strategy_desc

def predict_adaptive(df, target_period, window_size=30):
    """阶段性自适应预测"""
    positions = ['万位', '千位', '百位', '十位']

    mask = df['期号'] == target_period
    if not mask.any():
        return None, "期号不在数据中"

    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None, f"数据不足，只有{target_idx}期历史数据"

    window = df.iloc[target_idx - window_size:target_idx]

    result = {}
    regime_info = {}

    for pos in positions:
        regime, confidence, features = analyze_regime(window, pos)
        kill, strategy = get_kill_strategy(regime, features, window[pos])

        result[pos] = kill
        regime_info[pos] = {
            'regime': regime,
            'confidence': confidence,
            'features': features,
            'strategy': strategy
        }

    return result, regime_info

# ========== 界面 ==========
st.title("🎯 排列五阶段性自适应杀号系统")
st.markdown("#### 智能识别当前阶段，动态切换杀号策略（纯Python实现，零额外依赖）")

try:
    df = load_data()
    st.info(f"📊 已加载历史数据：{len(df)}期（{int(df['期号'].min())} - {int(df['期号'].max())}）")
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

# 创建标签页
tab1, tab2 = st.tabs(["🔮 自适应预测", "📊 阶段分析"])

with tab1:
    st.subheader("当前阶段识别与预测")

    col1, col2, col3 = st.columns(3)
    with col1:
        target_period = st.number_input(
            "目标期号",
            min_value=int(df['期号'].min()) + 30,
            max_value=int(df['期号'].max()) + 10,
            value=int(df['期号'].max()),
        )
    with col2:
        window_size = st.selectbox("观察窗口", options=[20, 30, 50], index=1)
    with col3:
        force_regime = st.selectbox(
            "强制阶段",
            options=["自动识别", "🔥 热号延续", "❄️ 冷号回补", "📈 连号趋势", "🎲 混沌无序"],
            index=0,
        )

    if st.button("🔮 开始自适应预测", type="primary", use_container_width=True):
        with st.spinner("正在分析当前阶段..."):
            result, regime_info = predict_adaptive(df, target_period, window_size)

        if result is None:
            st.error(regime_info)
        else:
            st.markdown("### 🎭 当前阶段识别")

            cols = st.columns(4)
            positions = ['万位', '千位', '百位', '十位']

            for i, pos in enumerate(positions):
                with cols[i]:
                    info = regime_info[pos]
                    regime = info['regime']
                    conf = info['confidence']

                    color_map = {
                        "HOT": ("🔥 热号延续", "#e74c3c"),
                        "COLD": ("❄️ 冷号回补", "#3498db"),
                        "TREND": ("📈 连号趋势", "#2ecc71"),
                        "CHAOS": ("🎲 混沌无序", "#95a5a6")
                    }
                    label, color = color_map.get(regime, ("未知", "#000000"))

                    st.markdown(f"""
                    <div style="background:{color};padding:15px;border-radius:10px;text-align:center;color:white;margin:10px 0;">
                        <div style="font-size:24px;font-weight:bold;">{pos}</div>
                        <div style="font-size:16px;">{label}</div>
                        <div style="font-size:12px;opacity:0.9;">置信度: {conf:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("查看策略详情"):
                        st.write(info['strategy'])
                        f = info['features']
                        st.write(f"- 最高频占比: {f['max_freq_ratio']:.1%}")
                        st.write(f"- 最低频占比: {f['min_freq_ratio']:.1%}")
                        st.write(f"- 等差趋势: {f['arithmetic_ratio']:.1%}")
                        st.write(f"- 熵值: {f['entropy']:.2f}")

            st.markdown("### 🎯 杀号结果")
            cols2 = st.columns(4)
            for i, pos in enumerate(positions):
                with cols2[i]:
                    kills = result[pos]
                    st.markdown(f"""
                    <div style="background:white;padding:20px;border-radius:10px;text-align:center;box-shadow:0 4px 6px rgba(0,0,0,0.1);margin:10px 0;border:2px solid #e74c3c;">
                        <div style="color:#667eea;font-size:20px;font-weight:bold;margin-bottom:15px;">{pos}</div>
                        <div>
                            <span style="display:inline-block;width:60px;height:60px;line-height:60px;background:#e74c3c;color:white;border-radius:50%;font-size:28px;font-weight:bold;margin:0 10px;">{kills[0]}</span>
                            <span style="display:inline-block;width:60px;height:60px;line-height:60px;background:#e74c3c;color:white;border-radius:50%;font-size:28px;font-weight:bold;margin:0 10px;">{kills[1]}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

with tab2:
    st.subheader("阶段历史分析")

    col1, col2 = st.columns(2)
    with col1:
        analysis_start = st.number_input("分析起始期号", 
                                        value=int(df['期号'].max()) - 99,
                                        min_value=int(df['期号'].min()))
    with col2:
        analysis_end = st.number_input("分析结束期号", 
                                      value=int(df['期号'].max()),
                                      max_value=int(df['期号'].max()))

    if st.button("📊 生成阶段演变数据", use_container_width=True):
        mask = (df['期号'] >= analysis_start) & (df['期号'] <= analysis_end)
        analysis_df = df[mask].copy()

        if len(analysis_df) < 30:
            st.error("分析区间至少需要30期数据")
        else:
            window_size = 20
            regimes_history = {pos: [] for pos in ['万位', '千位', '百位', '十位']}
            periods = []

            progress_bar = st.progress(0)
            for idx in range(window_size, len(analysis_df)):
                period = int(analysis_df.iloc[idx]['期号'])
                periods.append(period)
                window = analysis_df.iloc[idx-window_size:idx]

                for pos in ['万位', '千位', '百位', '十位']:
                    regime, _, _ = analyze_regime(window, pos)
                    val = {"HOT": 3, "COLD": 2, "TREND": 1, "CHAOS": 0}.get(regime, 0)
                    regimes_history[pos].append(val)

                progress_bar.progress((idx - window_size + 1) / (len(analysis_df) - window_size))

            progress_bar.empty()

            # 使用Streamlit原生表格展示（不使用matplotlib）
            chart_data = pd.DataFrame({
                '期号': periods,
                '万位': regimes_history['万位'],
                '千位': regimes_history['千位'],
                '百位': regimes_history['百位'],
                '十位': regimes_history['十位']
            })

            st.line_chart(chart_data.set_index('期号'), height=400)
            st.caption("图例：0=混沌, 1=趋势, 2=冷号, 3=热号")

            # 详细数据表
            with st.expander("查看详细阶段数据"):
                st.dataframe(chart_data, use_container_width=True)

            # 统计各阶段占比
            st.markdown("### 📈 阶段分布统计")
            for pos in ['万位', '千位', '百位', '十位']:
                regime_counts = Counter([{"HOT": "热号", "COLD": "冷号", "TREND": "趋势", "CHAOS": "混沌"}.get(
                    ["CHAOS","TREND","COLD","HOT"][v], "未知") for v in regimes_history[pos]])
                total = len(regimes_history[pos])

                cols = st.columns(4)
                cols[0].metric(f"{pos}-热号", f"{regime_counts.get('热号', 0)}期", 
                             f"{regime_counts.get('热号', 0)/total:.1%}")
                cols[1].metric(f"{pos}-冷号", f"{regime_counts.get('冷号', 0)}期", 
                             f"{regime_counts.get('冷号', 0)/total:.1%}")
                cols[2].metric(f"{pos}-趋势", f"{regime_counts.get('趋势', 0)}期", 
                             f"{regime_counts.get('趋势', 0)/total:.1%}")
                cols[3].metric(f"{pos}-混沌", f"{regime_counts.get('混沌', 0)}期", 
                             f"{regime_counts.get('混沌', 0)/total:.1%}")

st.markdown("---")
st.caption("⚠️ 免责声明：本系统基于阶段性模式识别，但彩票本质仍是随机事件，仅供学习研究使用")
