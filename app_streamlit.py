import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict

st.set_page_config(page_title="排列五杀号预测系统", page_icon="🎯", layout="centered")

@st.cache_data
def load_data():
    """加载数据并清洗"""
    df = pd.read_excel('排列五开奖历史.xlsx', skiprows=1)
    df.columns = ['期号', '万位', '千位', '百位', '十位']

    for col in ['期号', '万位', '千位', '百位', '十位']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 删除空值
    df = df.dropna()

    # ✅ 关键修复1：删除重复期号（保留最后一个，即最新的）
    before_drop = len(df)
    df = df.drop_duplicates(subset=['期号'], keep='last')
    after_drop = len(df)

    if before_drop != after_drop:
        st.warning(f"⚠️ 检测到并删除了 {before_drop - after_drop} 个重复期号")

    # ✅ 关键修复2：严格按期号升序排列并重置索引
    df = df.sort_values('期号', ascending=True).reset_index(drop=True)

    return df

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

    # 找到目标期号的索引
    mask = df['期号'] == target_period
    if not mask.any():
        return None, f"期号{target_period}不在数据中"

    target_idx = df[mask].index[0]

    if target_idx < window_size:
        return None, f"数据不足，期号{target_period}前只有{target_idx}期数据"

    # 严格取目标期号之前的window_size期
    start_idx = target_idx - window_size
    window = df.iloc[start_idx:target_idx].copy()

    # ✅ 严格验证：确保窗口内绝对没有目标期号
    if target_period in window['期号'].values:
        periods_in_window = window['期号'].tolist()
        return None, f"❌ 数据泄露错误：训练数据包含目标期号{target_period}。窗口期号：{periods_in_window}"

    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()

        # ML部分（无数据泄露）
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

            X_train = X[:-1]
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

        # 集成
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

    return result, None

# 界面
st.title("🎯 排列五杀号预测系统（最终修复版）")

try:
    df = load_data()
    st.info(f"📊 已加载历史数据：{len(df)}期（{int(df['期号'].min())} - {int(df['期号'].max())}）")
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    target_period = st.number_input("目标期号", min_value=2025000, max_value=2027000, value=2026072)
with col2:
    window_size = st.selectbox("数据窗口", options=[30, 50], index=0)

if st.button("🔮 开始预测杀号", type="primary"):
    with st.spinner("正在分析..."):
        result, error = predict_kill_numbers(df, target_period, window_size)

    if error:
        st.error(error)
    else:
        st.success(f"✅ 第 {target_period} 期杀号预测完成（数据安全，无泄露）")

        cols = st.columns(4)
        positions = ['万位', '千位', '百位', '十位']
        for i, pos in enumerate(positions):
            with cols[i]:
                kills = result[pos]
                st.markdown(f"""
                <div style="background:white;padding:20px;border-radius:10px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color:#667eea;font-size:20px;font-weight:bold;margin-bottom:15px;">{pos}</div>
                    <div>
                        <span style="display:inline-block;width:50px;height:50px;line-height:50px;background:#e74c3c;color:white;border-radius:50%;font-size:24px;font-weight:bold;margin:0 5px;">{kills[0]}</span>
                        <span style="display:inline-block;width:50px;height:50px;line-height:50px;background:#e74c3c;color:white;border-radius:50%;font-size:24px;font-weight:bold;margin:0 5px;">{kills[1]}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.caption("⚠️ 免责声明：本系统仅供娱乐和学习使用，彩票开奖具有随机性，不构成任何投注建议")
