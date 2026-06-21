"""
AIDI v3 — 多维AI能力指数
==============================
AIC = 综合能力 (向量+交互效应)
AIDI = 发展速度 (向量模长增量)

能力空间: 6个维度 + 跨维度交互乘数
"""

import json, math
from datetime import datetime
from pathlib import Path

# ── 六维能力定义 ─────────────────────────────────────────────
DIMS = [
    "intelligence",  # LLM推理/知识/编程 - "大脑"
    "multimodal",    # 看/听/说/生成 - "感官"
    "agent",         # 工具/规划/自主/编排 - "手脚"
    "programming",   # 代码生成/自我改进 - "自我复制"
    "knowledge",     # RAG/向量库/记忆/检索 - "图书馆"
    "ecosystem",     # API/成本/框架/开发者生态 - "基础设施"
]

DIM_NAMES_ZH = {
    "intelligence": "语言智力",
    "multimodal": "多模态感知",
    "agent": "智能体行动力",
    "programming": "编程自改进",
    "knowledge": "知识系统",
    "ecosystem": "生态基础设施",
}

# ── 交互效应系数 ─────────────────────────────────────────────
# synergy[i][j] = 维度i和j的交互强度 (0-1)
# 语义: "当两个维度都强时，综合效果超过单维度之和"
# 数学形式: 交互项 = synergy × min(vi,vj)/1000 × (vi×vj/1e6)
SYNERGY = {
    ("intelligence", "agent"): 0.35,       # 大脑+手脚 = 能自主做事
    ("intelligence", "multimodal"): 0.20,  # 大脑+感官 = 理解真实世界
    ("intelligence", "programming"): 0.30, # 大脑+编程 = AI造AI
    ("intelligence", "knowledge"): 0.25,   # 大脑+图书馆 = 专家系统
    ("agent", "programming"): 0.25,        # 手脚+编程 = 自我改进工具
    ("agent", "ecosystem"): 0.20,          # 手脚+基础设施 = 普惠AI
    ("agent", "knowledge"): 0.20,          # 手脚+知识 = 有记忆的行动
    ("multimodal", "programming"): 0.15,   # 感官+编程 = 视觉编程
    ("multimodal", "agent"): 0.20,         # 感官+手脚 = 机器人
    ("knowledge", "ecosystem"): 0.15,      # 知识+生态 = 企业级知识库
    ("programming", "ecosystem"): 0.15,    # 编程+生态 = 应用生态
}

def calc_dim_aic(dim_scores):
    """计算维度基础AIC = 加权和 (权重平等)"""
    return sum(dim_scores.values()) / len(dim_scores)

def calc_synergy(dim_scores):
    """计算跨维度交互效应 (0~500加成)"""
    total_synergy = 0
    for (d1, d2), s in SYNERGY.items():
        v1 = dim_scores.get(d1, 0)
        v2 = dim_scores.get(d2, 0)
        if v1 > 0 and v2 > 0:
            # 交互项: 受弱维度约束 × 乘积效应
            weak = min(v1, v2) / 1000.0
            product = (v1 * v2) / 1_000_000.0
            synergy_term = s * weak * product * 500
            total_synergy += synergy_term
    return total_synergy

def calc_aic_v3(dim_scores):
    """AIC = 基础能力 + 交互效应"""
    base = calc_dim_aic(dim_scores)
    synergy = calc_synergy(dim_scores)
    return round(base + synergy)

def calc_aidi_v3(prev_aic, curr_aic):
    """AIDI = AIC增量"""
    return curr_aic - prev_aic

# ── 逐期历史数据: 六维评分 (0-1000) ──────────────────────────
# 基于关键里程碑估算, 每条代表该半月的综合能力水平
# intelligence分 = LLM能力
# multimodal分 = 图像/视频/音频/语音识别/生成
# agent分 = 工具调用/推理规划/自主Agent/多Agent编排
# programming分 = 代码生成/代码理解/自我改进
# knowledge分 = RAG/向量库/记忆/检索/知识图谱
# ecosystem分 = API可用性/成本/框架成熟度/开发者社区/普及度

# ── 逐期历史数据: 从 curated/dim_scores.json 读取 ──
PERIOD_SIX_DIMS = json.load(open(
    Path(__file__).parent.parent / "data/curated/dim_scores.json",
    encoding="utf-8"
))["periods"]



def build_timeseries_v3():
    """构建完整时间序列 (含AIC/AIDI计算)"""
    periods = []
    prev_aic = None

    sorted_keys = sorted(PERIOD_SIX_DIMS.keys())
    
    for key in sorted_keys:
        raw = PERIOD_SIX_DIMS[key]
        # Extract dimension scores from the data structure
        if isinstance(raw, dict) and "scores" in raw:
            dims = {k: v for k, v in raw["scores"].items() if k in DIMS}
        else:
            dims = {k: v for k, v in raw.items() if k in DIMS}
        aic = calc_aic_v3(dims)
        
        if prev_aic is not None:
            aidi = calc_aidi_v3(prev_aic, aic)
        else:
            aidi = 100  # 基线速度 (用户设定)
        note = raw.get("note", "") if isinstance(raw, dict) else ""
        
        periods.append({
            "date": key,
            "aic": aic,
            "aidi": aidi,
            "dimensions": dims.copy(),
            "note": note,
        })
        prev_aic = aic

    return periods


def run_full_v3():
    """全量分析"""
    periods = build_timeseries_v3()
    base = periods[0]
    curr = periods[-1]

    print("=" * 65)
    print("AIDI v3 — 六维交互指数")
    print("=" * 65)
    print(f"基线: {base['date']}  AIC={base['aic']}  AIDI={base['aidi']}")
    print(f"当前: {curr['date']}  AIC={curr['aic']}  AIDI={curr['aidi']:+}")
    print(f"增长: +{curr['aic'] - base['aic']}点")
    print()
    print(f"{'日期':<14} {'AIC':>6} {'AIDI':>6} {'智力':>5} {'感官':>5} {'行动':>5} {'编程':>5} {'知识':>5} {'生态':>5}")
    print("-" * 65)

    # 关键期 (锚点或AIDI波动期)
    for p in periods:
        dims = p['dimensions']
        note = p.get('note', '')
        if note or p['aidi'] > 50 or abs(p['aidi']) < 15:
            i = dims.get('intelligence', 0)
            m = dims.get('multimodal', 0)
            a = dims.get('agent', 0)
            c = dims.get('programming', 0)
            k = dims.get('knowledge', 0)
            e = dims.get('ecosystem', 0)
            print(f"{p['date']:<14} {p['aic']:>6} {p['aidi']:>+6} {i:>5} {m:>5} {a:>5} {c:>5} {k:>5} {e:>5}")

    # 交互效应分析
    print()
    print("交互效应分析 (当前):")
    cur_dims = curr['dimensions']
    base_aic = calc_dim_aic(cur_dims)
    synergy = calc_synergy(cur_dims)
    print(f"  基础能力(六维平均): {base_aic:.0f}")
    print(f"  交互效应加成: +{synergy:.0f}")
    print(f"  AIC综合: {curr['aic']}")
    print(f"  交互占比: {synergy / curr['aic'] * 100:.1f}%")

    # 保存报告
    report = {
        "baseline": {"date": base["date"], "aic": base["aic"], "aidi": base["aidi"]},
        "current": {"date": curr["date"], "aic": curr["aic"], "aidi": curr["aidi"]},
        "aic_growth": curr["aic"] - base["aic"],
        "aic_growth_pct": round((curr["aic"] / base["aic"] - 1) * 100, 1),
        "current_dimensions": cur_dims,
        "base_aic": round(base_aic),
        "synergy_bonus": round(synergy),
        "synergy_pct": round(synergy / curr["aic"] * 100, 1),
        "total_periods": len(periods),
    }

    report_path = "reports/aidi_v3_report.json"
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存: {report_path}")

    return report


if __name__ == "__main__":
    run_full_v3()
