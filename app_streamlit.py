import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict

st.set_page_config(page_title="排列五杀号预测系统", page_icon="🎯", layout="centered")

@st.cache_data
def load_data():
    df = pd.read_excel('排列五开奖历史.xlsx', skiprows=1)
    df.columns = ['期号', '万位', '千位', '百位', '十位']
    for col in ['期号', '万位', '千位', '百位', '十位']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 严格排序并重置索引
    df = df.sort_values('期号', ascending=True).reset_index(drop=True)

    # 检查是否有重复期号
    dupes = df[df.duplicated(subset=['期号'], keep=False)]
    if len(dupes) > 0:
        st.warning(f"⚠️ 发现重复期号：{dupes['期号'].tolist()}")

    return df.dropna()

def calculate_features(window_data, position):
    numbers = window_data[position].astype(int).tolist()
    features = {}
    features['mean'] = np.mean(numbers)
    features['std'] = np.std(numbers)

    last_seen = {}
    for i, num in enumerate(reversed(numbers)):
        if num not in last_seen:
            last_seen[num] = i
    for num in range(10):
        features[f'miss_{num}'] = last_seen.get(num, len(numbers))

    counter = Counter(numbers)
    for num in range(10):
        features[f'freq_{num}'] = counter.get(num, 0) / len(numbers)

    if len(numbers) >= 5:
        features['trend'] = np.polyfit(range(5), numbers[-5:], 1)[0]
    else:
        features['trend'] = 0

    return features

def markov_predict(numbers, order=2):
    if len(numbers) < order + 1:
        return {i: 0.1 for i in range(10)}

    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(numbers) - order):
        state = tuple(numbers[i:i+order])
        next_num = numbers[i+order]
        transitions[state][next_num] += 1

    current_state = tuple(numbers[-order:])
    if current_state not in transitions:
        return {i: 0.1 for i in range(10)}

    counts = transitions[current_state]
    total = sum(counts.values())
    probs = {k: v/total for k, v in counts.items()}

    for i in range(10):
        if i not in probs:
            probs[i] = 0.01
    total = sum(probs.values())
    return {k: v/total for k, v in probs.items()}

def predict_kill_numbers(df, target_period, window_size=30):
    positions = ['万位', '千位', '百位', '十位']

    # 找到目标期号的位置
    mask = df['期号'] == target_period
    if not mask.any():
        return None, None, None, f"期号{target_period}不在数据中"

    target_idx = df[mask].index[0]

    if target_idx < window_size:
        return None, None, None, f"数据不足，期号{target_period}前只有{target_idx}期数据"

    # 关键：取target_idx之前的window_size期，不包含target_idx本身
    start_idx = target_idx - window_size
    window = df.iloc[start_idx:target_idx].copy()

    # 验证：确保窗口内没有目标期号
    if target_period in window['期号'].values:
        return None, None, None, f"❌ 数据泄露错误：训练数据包含目标期号{target_period}"

    # 记录详细信息
    debug_info = {
        'target_idx': target_idx,
        'start_idx': start_idx,
        'data_range': f"{int(window['期号'].min())} - {int(window['期号'].max())}",
        'periods_list': window['期号'].tolist(),
        'count': len(window)
    }

    result = {}
    details = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()

        # ML部分 - 修复数据泄露
        X, y = [], []
        for i in range(10, len(numbers)):
            sub_window = window.iloc[max(0, i-10):i]
            if len(sub_window) >= 5:
                feat = calculate_features(sub_window, pos)
                X.append(list(feat.values()))
                y.append(numbers[i])

        ml_probs = {i: 0.1 for i in range(10)}
        if len(X) > 5:
            X = np.array(X)
            y = np.array(y)
            scaler = StandardScaler()

            X_train = X[:-1]  # 排除最后一个用于训练的样本
            X_pred = X[-1:]

            if len(X_train) >= 5:
                X_train_scaled = scaler.fit_transform(X_train)
                X_pred_scaled = scaler.transform(X_pred)

                for num in range(10):
                    y_binary = (y[:-1] == num).astype(int)
                    if sum(y_binary) > 0:
                        rf = RandomForestClassifier(n_estimators=50, random_state=42)
                        rf.fit(X_train_scaled, y_binary)
                        prob = rf.predict_proba(X_pred_scaled)[0][1]
                        ml_probs[num] = prob

        # 其他算法
        markov_probs = markov_predict(numbers, order=2)
        counter = Counter(numbers)
        stat_probs = {i: 1 - (counter.get(i, 0) / len(numbers)) for i in range(10)}

        last_seen = {}
        for i, num in enumerate(reversed(numbers)):
            if num not in last_seen:
                last_seen[num] = i
        miss_probs = {i: last_seen.get(i, window_size) / window_size for i in range(10)}

        final_probs = {}
        for i in range(10):
            final_probs[i] = (
                0.4 * (1 - ml_probs[i]) +
                0.3 * (1 - markov_probs[i]) +
                0.2 * stat_probs[i] +
                0.1 * (1 - miss_probs[i])
            )

        sorted_nums = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        kill_nums = [int(sorted_nums[0][0]), int(sorted_nums[1][0])]
        result[pos] = kill_nums

        details[pos] = {
            'final_probs': final_probs,
            'last_number': numbers[-1],
            'last_period': int(window['期号'].iloc[-1])
        }

    return result, debug_info, details, None

# 界面
st.title("🎯 排列五杀号预测系统（数据泄露修复版）")

try:
    df = load_data()
    st.info(f"📊 已加载历史数据：{len(df)}期（{int(df['期号'].min())} - {int(df['期号'].max())}）")
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    target_period = st.number_input("目标期号", min_value=2025000, max_value=2027000, value=2026071)
with col2:
    window_size = st.selectbox("数据窗口", options=[30, 50], index=0)

if st.button("🔮 开始预测杀号", type="primary"):
    with st.spinner("正在分析..."):
        result, debug_info, details, error = predict_kill_numbers(df, target_period, window_size)

    if error:
        st.error(error)
    else:
        st.success(f"✅ 第 {target_period} 期杀号预测完成")

        # 显示关键调试信息
        with st.expander("🔍 关键调试信息（验证数据泄露）", expanded=True):
            st.write(f"**目标期号索引**：{debug_info['target_idx']}")
            st.write(f"**窗口起始索引**：{debug_info['start_idx']}")
            st.write(f"**训练数据范围**：{debug_info['data_range']}（共{debug_info['count']}期）")
            st.write(f"**窗口期号列表**：{debug_info['periods_list']}")

            if target_period in debug_info['periods_list']:
                st.error("🚨 警告：训练数据包含目标期号！数据泄露！")
            else:
                st.success("✅ 验证通过：训练数据不包含目标期号")

        # 显示结果
        cols = st.columns(4)
        positions = ['万位', '千位', '百位', '十位']
        for i, pos in enumerate(positions):
            with cols[i]:
                kills = result[pos]
                st.metric(label=pos, value=f"{kills[0]}, {kills[1]}")

        # 详细分析
        with st.expander("📈 详细分析"):
            for pos in positions:
                st.write(f"**{pos}**：")
                st.write(f"- 窗口最后一期（{details[pos]['last_period']}）实际号码：{details[pos]['last_number']}")
                st.write(f"- 杀号：{result[pos]}")

st.caption("⚠️ 免责声明：本系统仅供娱乐和学习使用")
