import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import traceback

st.set_page_config(page_title="排列五策略竞技场-调试版", page_icon="🐛", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('排列五开奖历史.xlsx', skiprows=1)
    df.columns = ['期号', '万位', '千位', '百位', '十位']
    for col in ['期号', '万位', '千位', '百位', '十位']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().drop_duplicates(subset=['期号'], keep='last')
    df = df.sort_values('期号', ascending=True).reset_index(drop=True)
    return df

def expert_ml_safe(df, target_period, window_size=30):
    """ML大师 - 带错误处理"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        st.error(f"❌ ML大师无法加载sklearn: {e}")
        return None, "sklearn未安装"

    positions = ['万位', '千位', '百位', '十位']
    mask = df['期号'] == target_period
    if not mask.any():
        return None, "期号不存在"

    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None, f"数据不足: 只有{target_idx}期 < 需要{window_size}期"

    window = df.iloc[target_idx - window_size:target_idx]
    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()

        features = []
        for i in range(10, len(numbers)):
            feat = [
                np.mean(numbers[max(0,i-10):i]),
                np.std(numbers[max(0,i-10):i]),
                numbers[i-1],
                numbers[i-2] if i >= 2 else 0,
            ]
            features.append(feat)

        if len(features) < 5:
            import random
            result[pos] = random.sample(range(10), 2)
        else:
            X = np.array(features[:-1])
            y = np.array(numbers[10:-1])

            if len(X) < 5:
                result[pos] = [numbers[-1], (numbers[-1]+1)%10]
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_pred = scaler.transform([features[-1]])

                probs = {}
                for num in range(10):
                    y_binary = (y == num).astype(int)
                    if sum(y_binary) > 0 and len(set(y_binary)) > 1:
                        rf = RandomForestClassifier(n_estimators=10, random_state=42)
                        rf.fit(X_scaled, y_binary)
                        prob = rf.predict_proba(X_pred)[0][1]
                        probs[num] = 1 - prob
                    else:
                        probs[num] = 0.5

                kill = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]
                result[pos] = [k[0] for k in kill]

    return result, "成功"

def expert_hot_debug(df, target_period, window_size=30):
    """热号猎手 - 带调试"""
    positions = ['万位', '千位', '百位', '十位']
    mask = df['期号'] == target_period

    if not mask.any():
        return None, "期号不存在"

    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None, f"数据不足: 索引{target_idx} < 窗口{window_size}"

    window = df.iloc[target_idx - window_size:target_idx]
    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()
        freq = Counter(numbers)
        cold = sorted(freq.items(), key=lambda x: x[1])[:2]
        result[pos] = [c[0] for c in cold]

    return result, f"成功，使用{window_size}期数据"

def expert_cold_debug(df, target_period, window_size=30):
    """冷号刺客 - 带调试"""
    positions = ['万位', '千位', '百位', '十位']
    mask = df['期号'] == target_period

    if not mask.any():
        return None, "期号不存在"

    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None, f"数据不足"

    window = df.iloc[target_idx - window_size:target_idx]
    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()
        freq = Counter(numbers)
        hot = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:2]
        result[pos] = [h[0] for h in hot]

    return result, "成功"

def expert_trend_debug(df, target_period, window_size=30):
    """趋势先知 - 带调试"""
    positions = ['万位', '千位', '百位', '十位']
    mask = df['期号'] == target_period

    if not mask.any():
        return None, "期号不存在"

    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None, f"数据不足"

    window = df.iloc[target_idx - window_size:target_idx]
    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()
        diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]

        if len(diffs) >= 3:
            diff_freq = Counter(diffs)
            common_diff = diff_freq.most_common(1)[0][0]
        else:
            common_diff = 0

        last = numbers[-1]
        predicted = (last + common_diff) % 10

        distances = {i: abs((i - predicted) % 10) for i in range(10)}
        kill = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:2]
        result[pos] = [k[0] for k in kill]

    return result, "成功"

def expert_chaos_debug(df, target_period, window_size=30):
    """混沌行者 - 带调试"""
    positions = ['万位', '千位', '百位', '十位']
    mask = df['期号'] == target_period

    if not mask.any():
        return None, "期号不存在"

    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None, f"数据不足"

    window = df.iloc[target_idx - window_size:target_idx]
    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()
        recent = numbers[-5:]
        freq = Counter(recent)
        hot_recent = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:2]
        kill = [h[0] for h in hot_recent]

        if len(kill) < 2:
            kill = numbers[-2:]

        result[pos] = kill

    return result, "成功"

def evaluate_expert_debug(df, expert_func, start_period, end_period, window_size=30):
    """带详细调试的评估函数"""
    positions = ['万位', '千位', '百位', '十位']
    results = []
    errors = []

    mask = (df['期号'] >= start_period) & (df['期号'] <= end_period)
    test_periods = df[mask]['期号'].tolist()

    if len(test_periods) == 0:
        return 0, 0, 0, [], ["没有找到测试期号"]

    for period in test_periods:
        try:
            kill_result, msg = expert_func(df, int(period), window_size)

            if kill_result is None:
                errors.append(f"期号{period}: {msg}")
                continue

            actual_row = df[df['期号'] == period]
            if len(actual_row) == 0:
                errors.append(f"期号{period}: 找不到实际开奖数据")
                continue

            actual = actual_row[positions].values[0].astype(int)

            # 详细记录
            detail = {
                'period': int(period),
                'actual': actual.tolist(),
                'kill': kill_result,
                'positions_correct': []
            }

            is_correct = True
            for i, pos in enumerate(positions):
                actual_num = int(actual[i])
                kill_nums = kill_result[pos]
                pos_correct = actual_num in kill_nums
                detail['positions_correct'].append({
                    'position': pos,
                    'actual': actual_num,
                    'kill': kill_nums,
                    'correct': pos_correct
                })
                if not pos_correct:
                    is_correct = False

            detail['is_correct'] = is_correct
            results.append(detail)

        except Exception as e:
            errors.append(f"期号{period}: 异常 - {str(e)}")
            continue

    if len(results) == 0:
        return 0, 0, 0, [], errors

    correct_count = sum([r['is_correct'] for r in results])
    total = len(results)
    accuracy = correct_count / total * 100

    return accuracy, correct_count, total, results, errors

# ========== 界面 ==========
st.title("🐛 策略竞技场-调试版（定位0%问题）")

try:
    df = load_data()
    st.info(f"📊 数据加载成功：{len(df)}期（{int(df['期号'].min())} - {int(df['期号'].max())}）")
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

st.markdown("---")

# 快速测试区域
st.subheader("🔍 快速诊断测试")

test_period = st.number_input("测试期号（用于单期诊断）", 
                             value=int(df['期号'].max()),
                             min_value=int(df['期号'].min()))

if st.button("🔬 运行单期诊断", type="primary"):
    st.write(f"正在测试期号：{test_period}")

    # 显示该期号的索引位置
    mask = df['期号'] == test_period
    if mask.any():
        idx = df[mask].index[0]
        st.write(f"- 该期号在数据中的索引：{idx}")
        st.write(f"- 前面可用的历史期数：{idx}")

        if idx < 30:
            st.error(f"❌ 数据不足！该期号前面只有{idx}期，不足30期窗口")
        else:
            st.success(f"✅ 数据充足，可以使用{idx}期历史数据")

            # 测试每个专家
            experts = {
                "🤖 ML大师": expert_ml_safe,
                "🔥 热号猎手": expert_hot_debug,
                "❄️ 冷号刺客": expert_cold_debug,
                "📈 趋势先知": expert_trend_debug,
                "🎲 混沌行者": expert_chaos_debug
            }

            for name, func in experts.items():
                st.write(f"---")
                st.write(f"**测试 {name}：**")
                result, msg = func(df, test_period, 30)

                if result is None:
                    st.error(f"❌ 失败：{msg}")
                else:
                    st.success(f"✅ {msg}")
                    st.write(f"预测结果：{result}")

                    # 显示实际开奖
                    actual = df[df['期号'] == test_period][['万位', '千位', '百位', '十位']].values[0]
                    st.write(f"实际开奖：万位{actual[0]}, 千位{actual[1]}, 百位{actual[2]}, 十位{actual[3]}")

                    # 验证每个位置
                    for i, pos in enumerate(['万位', '千位', '百位', '十位']):
                        is_killed = actual[i] in result[pos]
                        status = "✅ 杀中" if is_killed else "❌ 未中"
                        st.write(f"- {pos}: 实际{actual[i]}, 杀号{result[pos]} → {status}")

st.markdown("---")

# 批量评估区域
st.subheader("📊 批量评估（带详细错误日志）")

col1, col2, col3 = st.columns(3)
with col1:
    eval_start = st.number_input("评估起始", value=int(df['期号'].max()) - 20)
with col2:
    eval_end = st.number_input("评估结束", value=int(df['期号'].max()))
with col3:
    window_size = st.selectbox("窗口大小", [20, 30, 50], index=1)

if st.button("🚀 启动详细评估", type="primary"):
    experts = {
        "🤖 ML大师": expert_ml_safe,
        "🔥 热号猎手": expert_hot_debug,
        "❄️ 冷号刺客": expert_cold_debug,
        "📈 趋势先知": expert_trend_debug,
        "🎲 混沌行者": expert_chaos_debug
    }

    for name, func in experts.items():
        st.write(f"---")
        st.subheader(f"评估 {name}")

        with st.spinner(f"正在评估{name}..."):
            acc, correct, total, results, errors = evaluate_expert_debug(
                df, func, eval_start, eval_end, window_size
            )

        st.metric("总胜率", f"{acc:.1f}%", f"{correct}/{total}")

        if len(errors) > 0:
            with st.expander(f"查看错误日志 ({len(errors)}条)"):
                for err in errors[:10]:  # 只显示前10条
                    st.write(f"- {err}")

        if len(results) > 0:
            with st.expander(f"查看详细结果 ({len(results)}期)"):
                for r in results[:5]:  # 只显示前5期
                    st.write(f"期号{r['period']}: 开奖{r['actual']} → {'✅' if r['is_correct'] else '❌'}")
                    for pos_info in r['positions_correct']:
                        status = "✅" if pos_info['correct'] else "❌"
                        st.write(f"  {pos_info['position']}: 实际{pos_info['actual']} vs 杀号{pos_info['kill']} {status}")

st.markdown("---")
st.caption("调试版：用于定位为什么全是0%的问题")
