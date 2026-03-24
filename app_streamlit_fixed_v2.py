import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict

# 页面配置
st.set_page_config(
    page_title="排列五杀号预测系统",
    page_icon="🎯",
    layout="centered"
)

# CSS样式
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
    }
    .kill-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px;
    }
    .kill-number {
        display: inline-block;
        width: 50px;
        height: 50px;
        line-height: 50px;
        background: #e74c3c;
        color: white;
        border-radius: 50%;
        font-size: 24px;
        font-weight: bold;
        margin: 0 10px;
    }
    .position-title {
        color: #667eea;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .debug-info {
        background: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        font-size: 12px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """加载数据"""
    df = pd.read_excel('排列五开奖历史.xlsx', skiprows=1)
    df.columns = ['期号', '万位', '千位', '百位', '十位']
    for col in ['期号', '万位', '千位', '百位', '十位']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ✅ 确保按期号升序排列（从旧到新）
    df = df.sort_values('期号', ascending=True).reset_index(drop=True)

    return df.dropna()

def calculate_features(window_data, position):
    """计算特征"""
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
    """马尔可夫链预测"""
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
    """预测杀号 - 修复数据泄露版本"""
    positions = ['万位', '千位', '百位', '十位']

    target_idx = df[df['期号'] == target_period].index
    if len(target_idx) == 0:
        target_idx = len(df)
    else:
        target_idx = target_idx[0]

    if target_idx < window_size:
        return None, None, None  # ✅ 修复：返回3个值，防止解包错误

    start_idx = target_idx - window_size
    window = df.iloc[start_idx:target_idx]

    # 记录调试信息
    debug_info = {
        'data_range': f"{int(window['期号'].min())} - {int(window['期号'].max())}",
        'target': target_period,
        'samples': len(window)
    }

    result = {}
    details = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()

        # ✅ 修复数据泄露：确保训练集不包含最后一个样本
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

            # 关键修复：分离训练集(前19个)和预测集(最后1个)
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

        # 马尔可夫链
        markov_probs = markov_predict(numbers, order=2)

        # 统计频率
        counter = Counter(numbers)
        stat_probs = {i: 1 - (counter.get(i, 0) / len(numbers)) for i in range(10)}

        # 遗漏值
        last_seen = {}
        for i, num in enumerate(reversed(numbers)):
            if num not in last_seen:
                last_seen[num] = i
        miss_probs = {i: last_seen.get(i, window_size) / window_size for i in range(10)}

        # 集成预测
        final_probs = {}
        for i in range(10):
            final_probs[i] = (
                0.4 * (1 - ml_probs[i]) +
                0.3 * (1 - markov_probs[i]) +
                0.2 * stat_probs[i] +
                0.1 * (1 - miss_probs[i])
            )

        # 选择概率最高的2个作为杀号
        sorted_nums = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        kill_nums = [int(sorted_nums[0][0]), int(sorted_nums[1][0])]
        result[pos] = kill_nums

        # 保存详细概率供调试
        details[pos] = {
            'final_probs': final_probs,
            'ml': ml_probs,
            'markov': markov_probs,
            'actual_last': numbers[-1]
        }

    return result, debug_info, details

# 主界面
st.title("🎯 排列五杀号预测系统")

st.markdown("""
<div style="background:#e3f2fd;padding:15px;border-radius:10px;margin-bottom:20px;">
    💡 <b>使用说明</b>：输入目标期号，系统根据前30期或50期历史数据，
    融合<b>机器学习（随机森林）</b>+<b>马尔可夫链</b>+<b>时间序列分析</b>，
    每个位置杀2个最不可能出现的号码。
</div>
""", unsafe_allow_html=True)

# 加载数据
try:
    df = load_data()
    st.info(f"📊 已加载历史数据：{len(df)}期（{int(df['期号'].min())} - {int(df['期号'].max())}）")
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

# 输入区域
col1, col2 = st.columns(2)

with col1:
    target_period = st.number_input(
        "目标期号",
        min_value=2025000,
        max_value=2027000,
        value=2026072,
        help="输入要预测的期号，例如2026072"
    )

with col2:
    window_size = st.selectbox(
        "数据窗口",
        options=[30, 50],
        index=0,
        help="使用前几期数据进行预测"
    )

# 预测按钮
if st.button("🔮 开始预测杀号", type="primary"):
    with st.spinner("正在分析历史数据，融合多模型预测中..."):
        result, debug_info, details = predict_kill_numbers(df, target_period, window_size)

    if result is None:
        st.error(f"❌ 数据不足！期号{target_period}前没有{window_size}期历史数据")
    else:
        st.success(f"✅ 第 {target_period} 期杀号预测完成")

        # 显示数据范围调试用
        st.markdown(f"<div class='debug-info'>📅 训练数据范围：{debug_info['data_range']}（共{debug_info['samples']}期）→ 预测：{target_period}</div>", unsafe_allow_html=True)

        # 显示结果
        st.markdown("<br>", unsafe_allow_html=True)

        cols = st.columns(4)
        positions = ['万位', '千位', '百位', '十位']

        for i, pos in enumerate(positions):
            with cols[i]:
                kills = result[pos]
                st.markdown(f"""
                <div class="kill-box">
                    <div class="position-title">{pos}</div>
                    <div>
                        <span class="kill-number">{kills[0]}</span>
                        <span class="kill-number">{kills[1]}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 详细分析
        with st.expander("📈 查看详细分析（调试用）"):
            st.write(f"**目标期号**：{target_period}")
            st.write(f"**训练数据**：{debug_info['data_range']}（严格排除{target_period}期本身）")
            st.write("---")

            for pos in positions:
                st.write(f"**{pos}**：")
                st.write(f"- 窗口最后一期实际号码：{details[pos]['actual_last']}")
                st.write(f"- 各号码最终概率（越高越可能被杀）：")
                probs = details[pos]['final_probs']
                sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                st.write(f"  杀号：{sorted_p[0][0]}({sorted_p[0][1]:.3f}), {sorted_p[1][0]}({sorted_p[1][1]:.3f})")

            st.write("---")
            st.write("算法说明：")
            st.write("1. **随机森林**：基于前19期特征训练，预测第20期（修复数据泄露）")
            st.write("2. **马尔可夫链**：分析前2期状态转移概率")
            st.write("3. **遗漏分析**：统计号码遗漏值（冷号/热号）")
            st.write("4. **趋势分析**：近期走势回归分析")

st.markdown("---")
st.caption("⚠️ 免责声明：本系统仅供娱乐和学习使用，彩票开奖具有随机性，不构成任何投注建议。请理性购彩，量力而行。")
