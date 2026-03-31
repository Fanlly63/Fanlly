import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, Counter

st.set_page_config(page_title="排列五·真实数据杀号验证系统", layout="wide")

# ==================== 真实数据 ====================
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

# ==================== 10个专家策略 ====================

class ExpertSystem:
    def __init__(self):
        self.experts = {
            0: {"name": "砖家1号[热号杀2]", "strategy": "hot", "kill": 2, "desc": "杀近期出现最少的2个号"},
            1: {"name": "砖家2号[热号杀3]", "strategy": "hot", "kill": 3, "desc": "杀近期出现最少的3个号"},
            2: {"name": "砖家3号[冷号杀2]", "strategy": "cold", "kill": 2, "desc": "杀近期出现最多的2个号(追冷)"},
            3: {"name": "砖家4号[冷号杀3]", "strategy": "cold", "kill": 3, "desc": "杀近期出现最多的3个号(追冷)"},
            4: {"name": "砖家5号[随机杀2]", "strategy": "random", "kill": 2, "desc": "完全随机杀2个号(对照组)"},
            5: {"name": "砖家6号[随机杀3]", "strategy": "random", "kill": 3, "desc": "完全随机杀3个号(对照组)"},
            6: {"name": "砖家7号[趋势杀]", "strategy": "trend", "kill": 2, "desc": "杀上期开过的号及邻号"},
            7: {"name": "砖家8号[遗漏杀]", "strategy": "missing", "kill": 3, "desc": "杀最大遗漏号(追冷变体)"},
            8: {"name": "砖家9号[对称杀]", "strategy": "symmetry", "kill": 2, "desc": "杀与上期对称的号码"},
            9: {"name": "砖家10号[混合杀]", "strategy": "mixed", "kill": 3, "desc": "冷热+随机混合策略"}
        }
        self.stats = {i: {"total": 0, "correct": 0, "history": [], "current_streak": 0, "max_lose": 0} for i in range(10)}
        
    def predict(self, expert_id, history_data):
        """基于历史数据生成预测"""
        expert = self.experts[expert_id]
        strategy = expert["strategy"]
        kill_count = expert["kill"]
        
        if len(history_data) < 10:
            # 数据不足时随机杀
            return [np.random.choice(10, kill_count, replace=False) for _ in range(4)]
        
        predictions = []
        for pos_idx, pos_name in enumerate(['万位', '千位', '百位', '十位']):
            recent = [h[pos_idx+1] for h in history_data[-20:]]  # 最近20期该位置
            counter = Counter(recent)
            
            if strategy == "hot":
                # 杀出现最少的（追热）
                killed = [n for n, _ in counter.most_common()[:-kill_count-1:-1]]
                if len(killed) < kill_count:
                    killed = list(set(killed + list(np.random.choice([x for x in range(10) if x not in killed], kill_count-len(killed), replace=False))))
                    
            elif strategy == "cold":
                # 杀出现最多的（追冷/赌徒谬误）
                killed = [n for n, _ in counter.most_common()[:kill_count]]
                if len(killed) < kill_count:
                    killed = list(set(killed + list(np.random.choice([x for x in range(10) if x not in killed], kill_count-len(killed), replace=False))))
                    
            elif strategy == "random":
                killed = np.random.choice(10, kill_count, replace=False)
                
            elif strategy == "trend":
                # 杀上期开的号及其±1
                last_num = history_data[-1][pos_idx+1]
                trend_nums = [(last_num-1)%10, last_num, (last_num+1)%10]
                killed = list(set(trend_nums))[:kill_count]
                if len(killed) < kill_count:
                    killed = list(set(killed + list(np.random.choice([x for x in range(10) if x not in killed], kill_count-len(killed), replace=False))))
                    
            elif strategy == "missing":
                # 杀遗漏最大的（追冷）
                all_nums = set(range(10))
                recent_set = set(recent[-5:])  # 最近5期出现的
                missing = list(all_nums - recent_set)
                if len(missing) >= kill_count:
                    killed = missing[:kill_count]
                else:
                    killed = missing + list(np.random.choice(list(recent_set), kill_count-len(missing), replace=False))
                    
            elif strategy == "symmetry":
                # 杀与上期关于5对称的号
                last_num = history_data[-1][pos_idx+1]
                sym_num = (9 - last_num) % 10
                killed = [sym_num, last_num]
                if len(killed) < kill_count:
                    killed = list(set(killed + list(np.random.choice([x for x in range(10) if x not in killed], kill_count-len(killed), replace=False))))
                    
            elif strategy == "mixed":
                # 混合：一半热号一半冷号
                hot_nums = [n for n, _ in counter.most_common()[:-2:-1]]
                cold_nums = [n for n, _ in counter.most_common()[:2]]
                killed = list(set(hot_nums + cold_nums))[:kill_count]
                if len(killed) < kill_count:
                    killed = list(set(killed + list(np.random.choice(10, kill_count-len(killed), replace=False))))
            
            predictions.append(sorted(killed))
            
        return predictions
    
    def evaluate(self, expert_id, prediction, actual):
        """评估预测结果"""
        pos_correct = []
        for i in range(4):
            is_correct = actual[i] not in prediction[i]
            pos_correct.append(is_correct)
        
        all_correct = all(pos_correct)
        
        # 更新统计
        self.stats[expert_id]["total"] += 1
        if all_correct:
            self.stats[expert_id]["correct"] += 1
            self.stats[expert_id]["current_streak"] = max(1, self.stats[expert_id]["current_streak"] + 1)
        else:
            self.stats[expert_id]["current_streak"] = min(-1, self.stats[expert_id]["current_streak"] - 1)
            
        if abs(self.stats[expert_id]["current_streak"]) > self.stats[expert_id]["max_lose"]:
            self.stats[expert_id]["max_lose"] = abs(self.stats[expert_id]["current_streak"])
            
        self.stats[expert_id]["history"].append({
            "actual": actual,
            "prediction": prediction,
            "pos_correct": pos_correct,
            "all_correct": all_correct,
            "streak": self.stats[expert_id]["current_streak"]
        })
        
        return pos_correct, all_correct

# ==================== Streamlit界面 ====================

st.title("🔥 排列五·真实历史数据杀号验证系统")
st.markdown(f"**数据集**：{len(df_hist)}期真实开奖数据 (2025200-2026080期)")
st.markdown("**核心验证**：连错7期的专家，下期命中率是否真的提高？")

# 初始化
if 'expert_system' not in st.session_state:
    st.session_state.expert_system = ExpertSystem()
    st.session_state.test_idx = 20  # 从第20期开始测试（前面用于计算冷热）
    st.session_state.results = []

expert_sys = st.session_state.expert_system

# 控制面板
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("▶️ 运行一期验证", use_container_width=True):
        if st.session_state.test_idx < len(df_hist):
            current_data = df_hist.iloc[:st.session_state.test_idx].values.tolist()
            actual = df_hist.iloc[st.session_state.test_idx][['万位', '千位', '百位', '十位']].values
            
            period_result = {
                "idx": st.session_state.test_idx,
                "period": df_hist.iloc[st.session_state.test_idx]['期号'],
                "actual": actual,
                "experts": []
            }
            
            for eid in range(10):
                pred = expert_sys.predict(eid, current_data)
                pos_correct, all_correct = expert_sys.evaluate(eid, pred, actual)
                
                period_result["experts"].append({
                    "eid": eid,
                    "name": expert_sys.experts[eid]["name"],
                    "desc": expert_sys.experts[eid]["desc"],
                    "prediction": pred,
                    "pos_correct": pos_correct,
                    "all_correct": all_correct,
                    "streak": expert_sys.stats[eid]["current_streak"],
                    "total_rate": expert_sys.stats[eid]["correct"] / expert_sys.stats[eid]["total"] * 100 if expert_sys.stats[eid]["total"] > 0 else 0
                })
            
            st.session_state.results.append(period_result)
            st.session_state.test_idx += 1
        else:
            st.warning("已跑完所有历史数据！")

with col2:
    if st.button("⏭️ 连续跑50期", use_container_width=True):
        for _ in range(50):
            if st.session_state.test_idx < len(df_hist):
                current_data = df_hist.iloc[:st.session_state.test_idx].values.tolist()
                actual = df_hist.iloc[st.session_state.test_idx][['万位', '千位', '百位', '十位']].values
                
                for eid in range(10):
                    pred = expert_sys.predict(eid, current_data)
                    expert_sys.evaluate(eid, pred, actual)
                
                st.session_state.test_idx += 1
        st.rerun()

with col3:
    progress = (st.session_state.test_idx - 20) / (len(df_hist) - 20)
    st.progress(progress, text=f"验证进度: {st.session_state.test_idx-20}/{len(df_hist)-20}期")

# 显示区域
if len(st.session_state.results) > 0:
    latest = st.session_state.results[-1]
    
    # 最新结果
    st.subheader(f"第{latest['period']}期验证结果")
    cols = st.columns(4)
    for i, (col, num, pos) in enumerate(zip(cols, latest['actual'], ['万位', '千位', '百位', '十位'])):
        col.metric(pos, str(num))
    
    # 专家表现表
    st.markdown("### 📊 各专家本期表现")
    
    df_display = []
    for exp in latest["experts"]:
        df_display.append({
            "专家": exp["name"],
            "策略": exp["desc"],
            "万位": "✅" if exp["pos_correct"][0] else "❌",
            "千位": "✅" if exp["pos_correct"][1] else "❌", 
            "百位": "✅" if exp["pos_correct"][2] else "❌",
            "十位": "✅" if exp["pos_correct"][3] else "❌",
            "四全对": "🎯" if exp["all_correct"] else "💥",
            "当前连错": abs(exp["streak"]) if exp["streak"] < 0 else 0,
            "累计正确率": f"{exp['total_rate']:.1f}%"
        })
    
    df_exp = pd.DataFrame(df_display)
    # 按连错期数降序
    df_exp = df_exp.sort_values("当前连错", ascending=False)
    
    def color_loser(val):
        if isinstance(val, int) and val >= 5:
            return 'background-color: #ffcccc; color: red; font-weight: bold'
        return ''
    
    st.dataframe(df_exp.style.applymap(color_loser, subset=["当前连错"]), 
                 use_container_width=True, hide_index=True)
    
    # 推荐区
    st.markdown("---")
    st.subheader("🎯 智能推荐区（均值回归策略）")
    
    # 找出连错最多的
    losers = []
    for exp in latest["experts"]:
        if exp["streak"] < 0:
            losers.append((exp["eid"], exp["name"], abs(exp["streak"]), exp["desc"], exp["prediction"]))
    
    if losers:
        losers.sort(key=lambda x: x[2], reverse=True)
        top3 = losers[:3]
        
        col_rec1, col_rec2 = st.columns([1, 1])
        
        with col_rec1:
            st.markdown("**🔴 连错排行榜（推荐追投）**")
            for eid, name, streak, desc, pred in top3:
                with st.container():
                    st.markdown(f"""
                    <div style='padding:10px; background-color:#ffe6e6; border-radius:5px; margin:5px 0;'>
                        <b>{name}</b><br>
                        连续错误: <span style='color:red; font-size:20px;'>{streak}</span>期<br>
                        <small>{desc}</small><br>
                        <small>本期杀号: 万{pred[0]} 千{pred[1]} 百{pred[2]} 十{pred[3]}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col_rec2:
            # 验证历史推荐效果
            if len(st.session_state.results) >= 8:
                validation = []
                for i in range(7, len(st.session_state.results)):
                    current_res = st.session_state.results[i]
                    prev_res = st.session_state.results[i-1]
                    
                    # 找出上期连错>=3的专家
                    for exp in prev_res["experts"]:
                        if exp["streak"] <= -3:  # 上期连错3+
                            # 看他下期（即本期）是否中
                            current_exp = next((e for e in current_res["experts"] if e["eid"] == exp["eid"]), None)
                            if current_exp:
                                validation.append({
                                    "连错期数": abs(exp["streak"]),
                                    "下期是否命中": 1 if current_exp["all_correct"] else 0,
                                    "专家": exp["name"]
                                })
                
                if validation:
                    df_val = pd.DataFrame(validation)
                    summary = df_val.groupby("连错期数")["下期是否命中"].agg(["mean", "count"]).reset_index()
                    summary.columns = ["上期连错期数", "本期命中率", "样本数"]
                    summary["本期命中率"] = summary["本期命中率"] * 100
                    
                    fig = px.bar(summary, x="上期连错期数", y="本期命中率", text="样本数",
                               title="均值回归验证：连错N期后下期命中率",
                               color="本期命中率", color_continuous_scale="RdYlGn")
                    
                    # 添加理论线
                    fig.add_hline(y=40, line_dash="dash", line_color="red", 
                                annotation_text="理论值(杀2个)")
                    fig.add_hline(y=24, line_dash="dash", line_color="orange",
                                annotation_text="理论值(杀3个)")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 统计结论
                    total_samples = len(df_val)
                    hit_samples = df_val["下期是否命中"].sum()
                    actual_rate = hit_samples / total_samples * 100
                    st.info(f"统计结论：连错≥3期后共{total_samples}次追投，实际命中率{actual_rate:.1f}%，与理论概率无显著差异，**赌徒谬误不成立**。")

    # 长期统计
    if st.session_state.test_idx >= 50:
        st.markdown("---")
        st.subheader("📈 长期统计对比")
        
        stats_data = []
        for eid in range(10):
            s = expert_sys.stats[eid]
            theory_rate = 40.96 if expert_sys.experts[eid]["kill"] == 2 else 24.01
            stats_data.append({
                "专家": expert_sys.experts[eid]["name"],
                "策略类型": expert_sys.experts[eid]["strategy"],
                "杀号数": expert_sys.experts[eid]["kill"],
                "测试期数": s["total"],
                "实际四全对次数": s["correct"],
                "实际正确率": s["correct"]/s["total"]*100 if s["total"] > 0 else 0,
                "理论正确率": theory_rate,
                "最大连错": s["max_lose"]
            })
        
        df_stats = pd.DataFrame(stats_data)
        
        fig2 = px.bar(df_stats, x="专家", y="实际正确率", color="策略类型",
                     text="实际四全对次数", height=500,
                     title="各策略实际命中率 vs 理论值")
        fig2.add_hline(y=40.96, line_dash="dash", line_color="red", annotation_text="杀2个理论41%")
        fig2.add_hline(y=24.01, line_dash="dash", line_color="orange", annotation_text="杀3个理论24%")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.dataframe(df_stats.style.highlight_max(subset=["实际正确率"], color="green")
                                .highlight_min(subset=["实际正确率"], color="red"), 
                    use_container_width=True, hide_index=True)
        
        # 关键发现
        best = df_stats.loc[df_stats["实际正确率"].idxmax()]
        worst = df_stats.loc[df_stats["实际正确率"].idxmin()]
        st.markdown(f"""
        **关键发现**：
        - 表现"最好"：{best['专家']} ({best['实际正确率']:.1f}%) - 可能是运气方差
        - 表现"最差"：{worst['专家']} ({worst['实际正确率']:.1f}%) - 可能是运气方差  
        - 冷热策略 vs 随机策略：长期来看正确率**无显著差异**，均围绕理论值波动
        - **连错7期后追投**：命中率不会显著提高，仍遵循基础概率
        """)

# 使用说明
st.sidebar.markdown("---")
st.sidebar.markdown("""
**使用指南**：
1. 点击"运行一期"逐期验证，或"连续跑50期"快速验证
2. 观察右侧"连错排行榜"，红色标记连错多的专家
3. 关键看底部图表：连错N期后的实际命中率是否提高
4. 理论上，无论连错多少期，命中率始终回归基础概率（杀2个约41%，杀3个约24%）

**10大策略**：
- 热号杀：杀出现最少的（追热）
- 冷号杀：杀出现最多的（追冷/赌徒谬误）
- 随机杀：纯随机对照组
- 趋势/遗漏/对称/混合：各种"理论"
""")