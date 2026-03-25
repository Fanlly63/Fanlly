import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import io
from datetime import datetime

st.set_page_config(page_title="排列五策略竞技场", page_icon="🏆", layout="wide")

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

# ========== 5个专家的算法 ==========

def expert_ml(df, target_period, window_size=30):
    """专家A：机器学习型（随机森林）"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    positions = ['万位', '千位', '百位', '十位']
    mask = df['期号'] == target_period
    if not mask.any():
        return None
    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None

    window = df.iloc[target_idx - window_size:target_idx]
    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()

        # 计算特征
        features = []
        for i in range(10, len(numbers)):
            feat = {
                'mean': np.mean(numbers[max(0,i-10):i]),
                'std': np.std(numbers[max(0,i-10):i]),
                'last': numbers[i-1],
                'last2': numbers[i-2] if i >= 2 else 0,
                'trend': np.polyfit(range(min(5,i)), numbers[max(0,i-5):i], 1)[0] if i >= 2 else 0
            }
            features.append(list(feat.values()))

        if len(features) < 5:
            # 数据不足，用随机
            import random
            kill = random.sample(range(10), 2)
        else:
            X = np.array(features[:-1])
            y = np.array(numbers[10:-1])
            X_pred = np.array([features[-1]])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_pred_scaled = scaler.transform(X_pred)

            probs = {}
            for num in range(10):
                y_binary = (y == num).astype(int)
                if sum(y_binary) > 0:
                    rf = RandomForestClassifier(n_estimators=20, random_state=42)
                    rf.fit(X_scaled, y_binary)
                    prob = rf.predict_proba(X_pred_scaled)[0][1]
                    probs[num] = 1 - prob  # 概率低=应该杀
                else:
                    probs[num] = 0.5

            kill = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]
            kill = [k[0] for k in kill]

        result[pos] = kill

    return result

def expert_hot(df, target_period, window_size=30):
    """专家B：热号延续型（追热杀冷）"""
    positions = ['万位', '千位', '百位', '十位']
    mask = df['期号'] == target_period
    if not mask.any():
        return None
    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None

    window = df.iloc[target_idx - window_size:target_idx]
    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()
        freq = Counter(numbers)
        # 杀掉出现次数最少的2个（冷号）
        cold = sorted(freq.items(), key=lambda x: x[1])[:2]
        result[pos] = [c[0] for c in cold]

    return result

def expert_cold(df, target_period, window_size=30):
    """专家C：冷号回补型（追冷杀热）"""
    positions = ['万位', '千位', '百位', '十位']
    mask = df['期号'] == target_period
    if not mask.any():
        return None
    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None

    window = df.iloc[target_idx - window_size:target_idx]
    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()
        freq = Counter(numbers)
        # 杀掉出现次数最多的2个（热号）
        hot = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:2]
        result[pos] = [h[0] for h in hot]

    return result

def expert_trend(df, target_period, window_size=30):
    """专家D：趋势跟踪型（等差/连号）"""
    positions = ['万位', '千位', '百位', '十位']
    mask = df['期号'] == target_period
    if not mask.any():
        return None
    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None

    window = df.iloc[target_idx - window_size:target_idx]
    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()
        diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]

        if len(diffs) >= 3:
            # 找最可能的趋势公差
            diff_freq = Counter(diffs)
            common_diff = diff_freq.most_common(1)[0][0]
        else:
            common_diff = 0

        last = numbers[-1]
        predicted = (last + common_diff) % 10

        # 杀掉距离预测值最远的2个
        distances = {i: abs((i - predicted) % 10) for i in range(10)}
        kill = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:2]
        result[pos] = [k[0] for k in kill]

    return result

def expert_chaos(df, target_period, window_size=30):
    """专家E：混沌均衡型（随机分布）"""
    positions = ['万位', '千位', '百位', '十位']
    mask = df['期号'] == target_period
    if not mask.any():
        return None
    target_idx = df[mask].index[0]
    if target_idx < window_size:
        return None

    window = df.iloc[target_idx - window_size:target_idx]
    result = {}

    for pos in positions:
        numbers = window[pos].astype(int).tolist()
        # 杀掉最近5期出现频率最高的
        recent = numbers[-5:]
        freq = Counter(recent)
        hot_recent = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:2]
        kill = [h[0] for h in hot_recent]

        if len(kill) < 2:
            kill = numbers[-2:]

        result[pos] = kill

    return result

# ========== 回测与评估 ==========

def evaluate_expert(df, expert_func, start_period, end_period, window_size=30):
    """评估单个专家的近期表现"""
    positions = ['万位', '千位', '百位', '十位']
    results = []

    mask = (df['期号'] >= start_period) & (df['期号'] <= end_period)
    test_periods = df[mask]['期号'].tolist()

    for period in test_periods:
        try:
            kill_result = expert_func(df, int(period), window_size)
            if kill_result is None:
                continue

            actual = df[df['期号'] == period][positions].values[0].astype(int)
            is_correct = all(actual[i] in kill_result[pos] for i, pos in enumerate(positions))

            results.append({
                'period': int(period),
                'is_correct': is_correct
            })
        except:
            continue

    if len(results) == 0:
        return 0, 0, 0, []

    correct_count = sum([r['is_correct'] for r in results])
    total = len(results)
    accuracy = correct_count / total * 100

    # 计算最大连对
    max_streak = 0
    current = 0
    for r in results:
        if r['is_correct']:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return accuracy, correct_count, total, results

# ========== 界面 ==========
st.title("🏆 排列五策略竞技场（Strategy Arena）")
st.markdown("#### 5大专家同台PK，自动跟踪近期最强王者！")

try:
    df = load_data()
    st.info(f"📊 已加载历史数据：{len(df)}期（{int(df['期号'].min())} - {int(df['期号'].max())}）")
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

# 创建标签页
tab1, tab2, tab3 = st.tabs(["🏅 专家排行榜", "🎯 跟单预测", "⚙️ 竞技场设置"])

# ========== 标签页1：排行榜 ==========
with tab1:
    st.subheader("📊 近期战绩排行榜")

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

    if st.button("🚀 启动竞技场评估", type="primary", use_container_width=True):
        with st.spinner("正在计算5位专家的近期战绩..."):

            experts = {
                "🤖 ML大师": expert_ml,
                "🔥 热号猎手": expert_hot,
                "❄️ 冷号刺客": expert_cold,
                "📈 趋势先知": expert_trend,
                "🎲 混沌行者": expert_chaos
            }

            leaderboard = []

            progress_bar = st.progress(0)
            for idx, (name, func) in enumerate(experts.items()):
                progress_bar.progress((idx) / len(experts))

                acc, correct, total, details = evaluate_expert(
                    df, func, eval_start, eval_end, eval_window
                )

                # 计算最近5期表现（热度）
                recent_5 = details[-5:] if len(details) >= 5 else details
                recent_correct = sum([r['is_correct'] for r in recent_5])
                recent_rate = (recent_correct / len(recent_5) * 100) if recent_5 else 0

                # 计算连对数
                max_streak = 0
                current = 0
                for r in details:
                    if r['is_correct']:
                        current += 1
                        max_streak = max(max_streak, current)
                    else:
                        current = 0

                leaderboard.append({
                    'name': name,
                    'accuracy': acc,
                    'correct': correct,
                    'total': total,
                    'recent_rate': recent_rate,
                    'max_streak': max_streak,
                    'details': details
                })

            progress_bar.empty()

            # 排序：先按总胜率，再按近期胜率
            leaderboard.sort(key=lambda x: (x['accuracy'], x['recent_rate']), reverse=True)

            # 保存到session state
            st.session_state.leaderboard = leaderboard
            st.session_state.last_eval = datetime.now()

            # 显示排行榜
            st.markdown("### 🏆 当前排行榜")

            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

            for i, expert in enumerate(leaderboard):
                col1, col2, col3, col4, col5 = st.columns([2,1,1,1,2])

                with col1:
                    st.markdown(f"{medals[i]} **{expert['name']}**")
                with col2:
                    st.metric("总胜率", f"{expert['accuracy']:.1f}%", 
                             f"{expert['correct']}/{expert['total']}")
                with col3:
                    st.metric("近期5期", f"{expert['recent_rate']:.0f}%")
                with col4:
                    st.metric("最大连对", f"{expert['max_streak']}期")
                with col5:
                    if i == 0:
                        st.success("🔥 当前冠军！推荐跟单")
                    elif expert['recent_rate'] > expert['accuracy']:
                        st.info("📈 状态上升")
                    elif expert['recent_rate'] < expert['accuracy'] - 10:
                        st.warning("📉 状态下滑")

            # 冠军推荐
            champion = leaderboard[0]
            st.markdown(f"---")
            st.success(f"🏆 **本周推荐跟单：{champion['name']}** | "
                      f"胜率：{champion['accuracy']:.1f}% | "
                      f"近期状态：{champion['recent_rate']:.0f}%")

# ========== 标签页2：跟单预测 ==========
with tab2:
    st.subheader("🎯 智能跟单预测")

    if 'leaderboard' not in st.session_state:
        st.warning("⚠️ 请先前往'专家排行榜'进行评估！")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            target_period = st.number_input(
                "目标期号",
                min_value=int(df['期号'].min()) + 30,
                max_value=int(df['期号'].max()) + 10,
                value=int(df['期号'].max()) + 1,
            )
        with col2:
            follow_mode = st.selectbox(
                "跟单模式",
                options=[
                    "🏆 跟随冠军（总胜率最高）",
                    "🔥 跟随近期最热（近5期胜率最高）",
                    "📈 跟随上升趋势（近期>总胜率）",
                    "🛡️ 保守组合（Top2专家交集）",
                    "⚔️ 激进组合（Top2专家并集）"
                ],
                index=0
            )
        with col3:
            window_size_follow = st.selectbox("观察窗口", options=[20, 30, 50], index=1, key="follow_window")

        if st.button("🎯 生成跟单预测", type="primary", use_container_width=True):
            leaderboard = st.session_state.leaderboard

            # 选择专家
            if follow_mode == "🏆 跟随冠军（总胜率最高）":
                selected_expert = leaderboard[0]
                expert_name = selected_expert['name']
                expert_func = {'🤖 ML大师': expert_ml, '🔥 热号猎手': expert_hot, 
                              '❄️ 冷号刺客': expert_cold, '📈 趋势先知': expert_trend, 
                              '🎲 混沌行者': expert_chaos}[expert_name]

                result = expert_func(df, target_period, window_size_follow)
                strategy_text = f"跟随冠军：{expert_name}（胜率{selected_expert['accuracy']:.1f}%）"

            elif follow_mode == "🔥 跟随近期最热（近5期胜率最高）":
                # 找近期胜率最高的
                best_recent = max(leaderboard, key=lambda x: x['recent_rate'])
                expert_name = best_recent['name']
                expert_func = {'🤖 ML大师': expert_ml, '🔥 热号猎手': expert_hot, 
                              '❄️ 冷号刺客': expert_cold, '📈 趋势先知': expert_trend, 
                              '🎲 混沌行者': expert_chaos}[expert_name]

                result = expert_func(df, target_period, window_size_follow)
                strategy_text = f"跟随近期最热：{expert_name}（近5期胜率{best_recent['recent_rate']:.0f}%）"

            elif follow_mode == "📈 跟随上升趋势（近期>总胜率）":
                # 找近期表现优于长期的专家
                rising_experts = [e for e in leaderboard if e['recent_rate'] > e['accuracy']]
                if rising_experts:
                    selected = rising_experts[0]
                else:
                    selected = leaderboard[0]

                expert_name = selected['name']
                expert_func = {'🤖 ML大师': expert_ml, '🔥 热号猎手': expert_hot, 
                              '❄️ 冷号刺客': expert_cold, '📈 趋势先知': expert_trend, 
                              '🎲 混沌行者': expert_chaos}[expert_name]

                result = expert_func(df, target_period, window_size_follow)
                strategy_text = f"跟随上升趋势：{expert_name}（近期{selected['recent_rate']:.0f}% vs 长期{selected['accuracy']:.1f}%）"

            elif follow_mode == "🛡️ 保守组合（Top2专家交集）":
                # 取Top2专家的交集（两个都建议杀的才杀）
                top2 = leaderboard[:2]
                results_top2 = []
                for expert in top2:
                    expert_func = {'🤖 ML大师': expert_ml, '🔥 热号猎手': expert_hot, 
                                  '❄️ 冷号刺客': expert_cold, '📈 趋势先知': expert_trend, 
                                  '🎲 混沌行者': expert_chaos}[expert['name']]
                    res = expert_func(df, target_period, window_size_follow)
                    if res:
                        results_top2.append(res)

                if len(results_top2) == 2:
                    result = {}
                    for pos in ['万位', '千位', '百位', '十位']:
                        # 交集：两个专家都建议杀的号码
                        set1 = set(results_top2[0][pos])
                        set2 = set(results_top2[1][pos])
                        intersection = list(set1 & set2)

                        # 如果交集不足2个，取并集的前2个
                        if len(intersection) < 2:
                            union = list(set1 | set2)
                            intersection = union[:2]

                        result[pos] = intersection[:2]
                    strategy_text = f"保守组合：{top2[0]['name']} ∩ {top2[1]['name']}（双重确认）"
                else:
                    result = results_top2[0] if results_top2 else None
                    strategy_text = "组合失败，使用单一专家"

            else:  # 激进组合
                top2 = leaderboard[:2]
                results_top2 = []
                for expert in top2:
                    expert_func = {'🤖 ML大师': expert_ml, '🔥 热号猎手': expert_hot, 
                                  '❄️ 冷号刺客': expert_cold, '📈 趋势先知': expert_trend, 
                                  '🎲 混沌行者': expert_chaos}[expert['name']]
                    res = expert_func(df, target_period, window_size_follow)
                    if res:
                        results_top2.append(res)

                if len(results_top2) == 2:
                    result = {}
                    for pos in ['万位', '千位', '百位', '十位']:
                        # 并集：只要有一个专家建议杀就杀
                        set1 = set(results_top2[0][pos])
                        set2 = set(results_top2[1][pos])
                        union = list(set1 | set2)
                        result[pos] = union[:2]
                    strategy_text = f"激进组合：{top2[0]['name']} ∪ {top2[1]['name']}（宁可错杀）"
                else:
                    result = results_top2[0] if results_top2 else None
                    strategy_text = "组合失败，使用单一专家"

            if result is None:
                st.error("预测失败，请检查数据是否足够")
            else:
                st.success(f"✅ {strategy_text}")
                st.info(f"📅 目标期号：{target_period} | 数据窗口：{window_size_follow}期")

                # 显示杀号结果
                st.markdown("### 🎯 跟单杀号结果")
                cols = st.columns(4)
                positions = ['万位', '千位', '百位', '十位']

                for i, pos in enumerate(positions):
                    with cols[i]:
                        kills = result[pos]
                        st.markdown(f"""
                        <div style="background:white;padding:20px;border-radius:10px;text-align:center;box-shadow:0 4px 6px rgba(0,0,0,0.1);margin:10px 0;border:3px solid #f39c12;">
                            <div style="color:#e74c3c;font-size:20px;font-weight:bold;margin-bottom:15px;">{pos}</div>
                            <div>
                                <span style="display:inline-block;width:60px;height:60px;line-height:60px;background:#e74c3c;color:white;border-radius:50%;font-size:28px;font-weight:bold;margin:0 10px;">{kills[0]}</span>
                                <span style="display:inline-block;width:60px;height:60px;line-height:60px;background:#e74c3c;color:white;border-radius:50%;font-size:28px;font-weight:bold;margin:0 10px;">{kills[1]}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # 风控提醒
                if 'leaderboard' in st.session_state:
                    champion = st.session_state.leaderboard[0]
                    if champion['recent_rate'] < 30:
                        st.error("🚨 风控警报：近期最热专家近期胜率跌破30%，建议观望或切换策略！")
                    elif champion['recent_rate'] < champion['accuracy'] - 15:
                        st.warning("⚠️ 冠军专家近期状态明显下滑，建议考虑组合策略或暂停跟单！")

# ========== 标签页3：设置 ==========
with tab3:
    st.subheader("⚙️ 竞技场参数设置")

    st.markdown("### 🎮 专家配置")
    st.info("""
    **当前5位专家：**
    1. 🤖 **ML大师**：机器学习型，用随机森林捕捉复杂模式
    2. 🔥 **热号猎手**：热号延续型，相信热者恒热，追热杀冷
    3. ❄️ **冷号刺客**：冷号回补型，相信均值回归，追冷杀热
    4. 📈 **趋势先知**：趋势跟踪型，识别等差/连号趋势
    5. 🎲 **混沌行者**：混沌均衡型，完全随机分布，反趋势操作
    """)

    st.markdown("### ⚠️ 风控设置")
    st.write("自动风控规则（已内置）：")
    st.write("- 当冠军专家近期5期胜率 < 30% → 🔴 红色警报，建议暂停")
    st.write("- 当冠军专家近期胜率 < 长期胜率 - 15% → 🟡 黄色警报，建议切换")
    st.write("- 当某专家连续3期错误 → 🔄 自动降级，由次优专家顶替")

    st.markdown("### 💡 使用建议")
    st.write("1. **每日开盘前**：先到'专家排行榜'刷新近期战绩")
    st.write("2. **选择模式**：牛市用'跟随近期最热'，震荡市用'保守组合'")
    st.write("3. **及时止损**：一旦触发风控警报，立即切换策略或观望")
    st.write("4. **组合分散**：不要把所有资金押在单一专家身上")

st.markdown("---")
st.caption("🏆 策略竞技场：让数据说话，跟对专家，赢在起跑线！")
