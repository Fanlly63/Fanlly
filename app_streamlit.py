import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import io

st.set_page_config(page_title="排列五杀号预测系统", page_icon="🎯", layout="centered")

@st.cache_data
def load_data():
    """加载数据并清洗"""
    df = pd.read_excel('排列五开奖历史.xlsx', skiprows=1)
    df.columns = ['期号', '万位', '千位', '百位', '十位']

    for col in ['期号', '万位', '千位', '百位', '十位']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    # 删除重复期号（保留最后一个）
    before_drop = len(df)
    df = df.drop_duplicates(subset=['期号'], keep='last')
    after_drop = len(df)

    # 严格排序
    df = df.sort_values('期号', ascending=True).reset_index(drop=True)

    return df, before_drop - after_drop

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
    """
    预测杀号 - 支持未来期号预测（数据中不存在的期号）
    """
    positions = ['万位', '千位', '百位', '十位']

    # ✅ 修复：检查目标期号是否在数据中
    mask = df['期号'] == target_period

    if mask.any():
        # 情况1：预测历史期号（回测）
        target_idx = df[mask].index[0]
        mode = "history"
    else:
        # 情况2：预测未来期号（最新数据+1）
        # 使用最后一期之后的位置
        target_idx = len(df)
        mode = "future"

    if target_idx < window_size:
        return None, f"数据不足，需要{window_size}期历史数据，但只有{target_idx}期"

    # 取前window_size期
    start_idx = target_idx - window_size
    window = df.iloc[start_idx:target_idx].copy()

    # 验证：如果是历史回测，确保不包含目标期号
    if mode == "history" and target_period in window['期号'].values:
        return None, f"❌ 数据泄露：训练数据包含目标期号{target_period}"

    result = {}
    debug_info = {
        'mode': mode,
        'target_idx': target_idx,
        'data_range': f"{int(window['期号'].min())} - {int(window['期号'].max())}",
        'count': len(window)
    }

    for pos in positions:
        numbers = window[pos].astype(int).tolist()

        # ML部分
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

    return result, debug_info

# ========== 界面开始 ==========
st.title("🎯 排列五杀号预测系统（完整版）")

# 初始化session state
if 'custom_data' not in st.session_state:
    st.session_state.custom_data = []

# 加载基础数据
try:
    df_base, dup_count = load_data()

    # 合并用户添加的数据
    if st.session_state.custom_data:
        df_custom = pd.DataFrame(st.session_state.custom_data)
        df = pd.concat([df_base, df_custom], ignore_index=True)
        df = df.drop_duplicates(subset=['期号'], keep='last')
        df = df.sort_values('期号', ascending=True).reset_index(drop=True)
    else:
        df = df_base

    st.info(f"📊 已加载历史数据：{len(df)}期（{int(df['期号'].min())} - {int(df['期号'].max())}）")
    if dup_count > 0:
        st.warning(f"⚠️ 原始数据中发现并清理了{dup_count}个重复期号")

except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

# ========== 数据更新功能区 ==========
with st.expander("➕ 添加新期号数据（每日更新）", expanded=False):
    st.markdown("当有新开奖结果时，在此添加：")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        new_period = st.number_input("期号", min_value=2025000, max_value=2028000, 
                                   value=int(df['期号'].max())+1)
    with col2:
        new_wan = st.number_input("万位", min_value=0, max_value=9, value=0)
    with col3:
        new_qian = st.number_input("千位", min_value=0, max_value=9, value=0)
    with col4:
        new_bai = st.number_input("百位", min_value=0, max_value=9, value=0)
    with col5:
        new_shi = st.number_input("十位", min_value=0, max_value=9, value=0)

    if st.button("✅ 添加到数据集", use_container_width=True):
        # 检查是否已存在
        if new_period in df['期号'].values:
            st.warning(f"期号{new_period}已存在，将更新数据")
            # 删除旧的
            st.session_state.custom_data = [d for d in st.session_state.custom_data if d['期号'] != new_period]

        # 添加新数据
        new_data = {
            '期号': int(new_period),
            '万位': int(new_wan),
            '千位': int(new_qian),
            '百位': int(new_bai),
            '十位': int(new_shi)
        }
        st.session_state.custom_data.append(new_data)
        st.success(f"✅ 已添加期号{new_period}：{new_wan}{new_qian}{new_bai}{new_shi}")
        st.rerun()

    # 显示已添加的临时数据
    if st.session_state.custom_data:
        st.markdown("**本次会话已添加的数据：**")
        for d in st.session_state.custom_data:
            st.write(f"期号{d['期号']}：{d['万位']}{d['千位']}{d['百位']}{d['十位']}")

        if st.button("🗑️ 清空本次添加的数据"):
            st.session_state.custom_data = []
            st.rerun()

    st.info("💡 提示：添加的数据仅在本次会话有效，如需永久保存请更新Excel文件")

# ========== 预测功能区 ==========
st.markdown("---")
st.subheader("🔮 杀号预测")

col1, col2 = st.columns(2)
with col1:
    # 默认值为最新期号+1（预测下一期）
    default_target = int(df['期号'].max()) + 1
    target_period = st.number_input("目标期号", min_value=2025000, max_value=2028000, 
                                   value=default_target,
                                   help="输入要预测的期号，可以是未来未开奖的期号")
with col2:
    window_size = st.selectbox("数据窗口", options=[30, 50], index=0)

if st.button("🔮 开始预测杀号", type="primary", use_container_width=True):
    with st.spinner("正在分析..."):
        result, debug_info = predict_kill_numbers(df, target_period, window_size)

    if result is None:
        st.error(debug_info)
    else:
        # 判断是回测还是预测
        if debug_info['mode'] == 'future':
            st.success(f"✅ 第 {target_period} 期（未来预测）杀号结果")
            st.info(f"📅 基于历史数据：{debug_info['data_range']}（最近{debug_info['count']}期）")
        else:
            st.success(f"✅ 第 {target_period} 期（历史回测）杀号结果")
            st.info(f"📅 训练数据范围：{debug_info['data_range']}（严格排除当期）")

        # 显示结果
        cols = st.columns(4)
        positions = ['万位', '千位', '百位', '十位']
        for i, pos in enumerate(positions):
            with cols[i]:
                kills = result[pos]
                st.markdown(f"""
                <div style="background:white;padding:20px;border-radius:10px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.1);margin:10px 0;">
                    <div style="color:#667eea;font-size:20px;font-weight:bold;margin-bottom:15px;">{pos}</div>
                    <div>
                        <span style="display:inline-block;width:50px;height:50px;line-height:50px;background:#e74c3c;color:white;border-radius:50%;font-size:24px;font-weight:bold;margin:0 5px;">{kills[0]}</span>
                        <span style="display:inline-block;width:50px;height:50px;line-height:50px;background:#e74c3c;color:white;border-radius:50%;font-size:24px;font-weight:bold;margin:0 5px;">{kills[1]}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ 免责声明：本系统仅供娱乐和学习使用，彩票开奖具有随机性，不构成任何投注建议。请理性购彩，量力而行。")
