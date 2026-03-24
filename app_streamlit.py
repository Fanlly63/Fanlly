import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import io
from datetime import datetime

st.set_page_config(page_title="排列五杀号预测系统", page_icon="🎯", layout="wide")

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
    """预测杀号"""
    positions = ['万位', '千位', '百位', '十位']

    mask = df['期号'] == target_period
    if not mask.any():
        return None, "期号不在数据中"

    target_idx = df[mask].index[0]

    if target_idx < window_size:
        return None, f"数据不足，期号{target_period}前只有{target_idx}期数据"

    start_idx = target_idx - window_size
    window = df.iloc[start_idx:target_idx].copy()

    # 验证：确保窗口内绝对没有目标期号
    if target_period in window['期号'].values:
        periods_in_window = window['期号'].tolist()
        return None, f"❌ 数据泄露：训练数据包含目标期号{target_period}"

    result = {}
    debug_info = {
        'target_idx': target_idx,
        'start_idx': start_idx,
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

        sorted_nums = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        kill_nums = [int(sorted_nums[0][0]), int(sorted_nums[1][0])]
        result[pos] = kill_nums

    return result, debug_info

def batch_backtest(df, start_period, end_period, window_size):
    """批量回测"""
    positions = ['万位', '千位', '百位', '十位']
    results = []

    # 获取回测期号列表
    mask = (df['期号'] >= start_period) & (df['期号'] <= end_period)
    test_periods = df[mask]['期号'].tolist()

    if len(test_periods) == 0:
        return None, "没有找到该区间内的期号数据"

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, period in enumerate(test_periods):
        status_text.text(f"正在回测第 {period} 期 ({i+1}/{len(test_periods)})...")
        progress_bar.progress((i + 1) / len(test_periods))

        # 预测
        kill_result, debug_info = predict_kill_numbers(df, int(period), window_size)

        if kill_result is None:
            continue

        # 获取实际开奖
        actual_row = df[df['期号'] == period]
        actual = actual_row[positions].values[0].astype(int)

        # 判断每个位置
        position_correct = {}
        for j, pos in enumerate(positions):
            actual_num = int(actual[j])
            kill_nums = kill_result[pos]
            is_killed = actual_num in kill_nums
            position_correct[pos] = is_killed

        # 4个位置全对才算当期正确
        is_correct = all(position_correct.values())

        results.append({
            '期号': int(period),
            '万位': actual[0],
            '千位': actual[1],
            '百位': actual[2],
            '十位': actual[3],
            '万位杀号': f"{kill_result['万位'][0]},{kill_result['万位'][1]}",
            '千位杀号': f"{kill_result['千位'][0]},{kill_result['千位'][1]}",
            '百位杀号': f"{kill_result['百位'][0]},{kill_result['百位'][1]}",
            '十位杀号': f"{kill_result['十位'][0]},{kill_result['十位'][1]}",
            '万位对': '✓' if position_correct['万位'] else '✗',
            '千位对': '✓' if position_correct['千位'] else '✗',
            '百位对': '✓' if position_correct['百位'] else '✗',
            '十位对': '✓' if position_correct['十位'] else '✗',
            '全对': '✓' if is_correct else '✗'
        })

    progress_bar.empty()
    status_text.empty()

    if len(results) == 0:
        return None, "回测失败，没有有效结果"

    return results, None

def analyze_consecutive(results):
    """分析连对连错"""
    correct_list = [r['全对'] == '✓' for r in results]

    max_consecutive_correct = 0
    max_consecutive_wrong = 0
    current_correct = 0
    current_wrong = 0
    consecutive_correct_periods = []
    consecutive_wrong_periods = []
    temp_periods = []

    for i, is_correct in enumerate(correct_list):
        period = results[i]['期号']
        if is_correct:
            current_correct += 1
            current_wrong = 0
            temp_periods.append(period)
            if current_correct > max_consecutive_correct:
                max_consecutive_correct = current_correct
                consecutive_correct_periods = temp_periods.copy()
            temp_periods = temp_periods[-max_consecutive_correct:]  # 保留最近的最大连对
        else:
            current_wrong += 1
            current_correct = 0
            temp_periods.append(period)
            if current_wrong > max_consecutive_wrong:
                max_consecutive_wrong = current_wrong
                consecutive_wrong_periods = temp_periods.copy()
            temp_periods = temp_periods[-max_consecutive_wrong:]

    return {
        'max_correct': max_consecutive_correct,
        'max_wrong': max_consecutive_wrong,
        'correct_periods': consecutive_correct_periods if max_consecutive_correct > 0 else [],
        'wrong_periods': consecutive_wrong_periods if max_consecutive_wrong > 0 else []
    }

# ========== 界面 ==========
st.title("🎯 排列五杀号预测系统（批量回测版）")

# 加载数据
try:
    df, dup_count = load_data()
    st.info(f"📊 已加载历史数据：{len(df)}期（{int(df['期号'].min())} - {int(df['期号'].max())}）")
    if dup_count > 0:
        st.warning(f"⚠️ 发现并清理了{dup_count}个重复期号")
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

# 创建标签页
tab1, tab2 = st.tabs(["🔮 单期预测", "📊 批量回测"])

# ========== 标签页1：单期预测 ==========
with tab1:
    st.subheader("单期杀号预测")

    col1, col2 = st.columns(2)
    with col1:
        target_period = st.number_input(
            "目标期号",
            min_value=2025000,
            max_value=2028000,
            value=int(df['期号'].max()) + 1,
            help="输入要预测的期号（可以是未来未开奖的期号）"
        )
    with col2:
        window_size_single = st.selectbox(
            "数据窗口（单期）",
            options=[30, 50],
            index=0,
            key="window_single"
        )

    if st.button("🔮 开始预测杀号", type="primary", use_container_width=True):
        with st.spinner("正在分析..."):
            result, debug_info = predict_kill_numbers(df, target_period, window_size_single)

        if result is None:
            st.error(debug_info)
        else:
            st.success(f"✅ 第 {target_period} 期杀号预测完成")
            st.info(f"📅 基于历史数据：{debug_info['data_range']}（共{debug_info['count']}期）")

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

# ========== 标签页2：批量回测 ==========
with tab2:
    st.subheader("📊 批量回测统计")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_period = st.number_input(
            "起始期号",
            min_value=int(df['期号'].min()),
            max_value=int(df['期号'].max()),
            value=2025230,
            help="回测开始的期号"
        )
    with col2:
        end_period = st.number_input(
            "结束期号",
            min_value=int(df['期号'].min()),
            max_value=int(df['期号'].max()),
            value=int(df['期号'].max()),
            help="回测结束的期号"
        )
    with col3:
        window_size_batch = st.selectbox(
            "数据窗口（回测）",
            options=[30, 50],
            index=0,
            key="window_batch",
            help="使用多少期历史数据进行预测"
        )

    if st.button("🚀 开始批量回测", type="primary", use_container_width=True):
        if start_period > end_period:
            st.error("❌ 起始期号不能大于结束期号")
        else:
            with st.spinner(f"正在回测 {start_period} 到 {end_period} 期，请稍候..."):
                results, error = batch_backtest(df, start_period, end_period, window_size_batch)

            if error:
                st.error(error)
            else:
                # 统计结果
                total_count = len(results)
                correct_count = sum([r['全对'] == '✓' for r in results])
                accuracy = correct_count / total_count * 100

                # 连对连错分析
                consecutive = analyze_consecutive(results)

                # 显示统计卡片
                st.markdown("### 📈 回测统计结果")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("总期数", f"{total_count}期")
                col2.metric("正确期数", f"{correct_count}期", f"{accuracy:.1f}%")
                col3.metric("最大连对", f"{consecutive['max_correct']}期")
                col4.metric("最大连错", f"{consecutive['max_wrong']}期")

                # 显示连对连错详情
                if consecutive['max_correct'] > 0:
                    st.success(f"✅ 最大连对 {consecutive['max_correct']} 期：期号 {consecutive['correct_periods'][0]} 到 {consecutive['correct_periods'][-1]}")
                if consecutive['max_wrong'] > 0:
                    st.error(f"❌ 最大连错 {consecutive['max_wrong']} 期：期号 {consecutive['wrong_periods'][0]} 到 {consecutive['wrong_periods'][-1]}")

                # 显示详细表格
                st.markdown("### 📋 详细回测结果")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)

                # 导出Excel按钮
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # 汇总表
                    summary_data = {
                        '指标': ['数据窗口', '回测区间', '总期数', '正确期数', '正确率(%)', '最大连对', '最大连错'],
                        '数值': [
                            f"{window_size_batch}期",
                            f"{start_period}-{end_period}",
                            total_count,
                            correct_count,
                            f"{accuracy:.2f}",
                            consecutive['max_correct'],
                            consecutive['max_wrong']
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='汇总', index=False)

                    # 详细结果表
                    df_results.to_excel(writer, sheet_name='详细结果', index=False)

                output.seek(0)
                st.download_button(
                    label="📥 下载回测结果Excel",
                    data=output,
                    file_name=f"回测结果_{start_period}_{end_period}_{window_size_batch}期_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

st.markdown("---")
st.caption("⚠️ 免责声明：本系统仅供娱乐和学习使用，彩票开奖具有随机性，不构成任何投注建议。请理性购彩，量力而行。")
