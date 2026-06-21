"""
AIDI v7 — 一阶齐次AIC指数引擎 (Cobb-Douglas + 归一化交互)
============================================================

设计目标:
  - 一阶齐次: 所有维度×k → AIC×k (解决v3 223倍膨胀问题)
  - 数学自洽: 增长倍数可解释, 交互效应不爆炸
  - 与经典指数接轨: Cobb-Douglas / HDI / CES 形式

公式:
  AIC = A × [Π dim_i^α_i] × [1 + η × Σ s_ij × R(dim_i, dim_j)]

  R(v_i, v_j) = min(v_i, v_j)² / (v_i² + v_j²)  ← 0次齐次归一化

验证步骤:
  1. 验证R函数在[0,0.5]内, 不随量级爆炸
  2. 纯几何平均(α_i=1/6, η=0)试跑
  3. 开启交互, 调η
  4. PSO外部基准拟合
  5. 回测对比v3
"""

import json, math
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent.parent
DIMS_FILE = BASE_DIR / "data/curated/dim_scores.json"

DIMS = ["intelligence", "multimodal", "agent", "programming", "knowledge", "ecosystem"]
DIM_NAMES_ZH = {
    "intelligence": "语言智力", "multimodal": "多模态感知",
    "agent": "智能体行动力", "programming": "编程自改进",
    "knowledge": "知识系统", "ecosystem": "生态基础设施",
}

# ── V7交互矩阵 (与v3同结构, 但R函数归一化) ──
SYNERGY_V7 = {
    ("intelligence", "agent"): 0.35,
    ("intelligence", "multimodal"): 0.20,
    ("intelligence", "programming"): 0.30,
    ("intelligence", "knowledge"): 0.25,
    ("agent", "programming"): 0.25,
    ("agent", "ecosystem"): 0.20,
    ("agent", "knowledge"): 0.20,
    ("multimodal", "programming"): 0.15,
    ("multimodal", "agent"): 0.20,
    ("knowledge", "ecosystem"): 0.15,
    ("programming", "ecosystem"): 0.15,
}


# ═══════════════════════════════════════════════════════════
#  核心函数
# ═══════════════════════════════════════════════════════════

def R_normalized(v_i, v_j):
    """0次齐次交互函数 R(v_i, v_j) = min² / (v_i² + v_j²)
    性质: [0, 0.5], (λv_i, λv_j)下不变"""
    if v_i <= 0 or v_j <= 0:
        return 0.0
    return (min(v_i, v_j) ** 2) / (v_i ** 2 + v_j ** 2)


def calc_aic_v7(dim_scores, alpha=None, eta=1.0, synergy=None, A=1.0):
    """AIC = A × CD_base × (1 + η × Σ s_ij × R_ij)
    
    Args:
        dim_scores: {dim: score} 原始六维评分
        alpha: 维度弹性, Σα=1, None=等权1/6
        eta: 交互总强度, 0=纯CD
        synergy: 交互强度矩阵, None=SYNERGY_V7
        A: 校准常数
    
    Returns:
        (aic_value, cd_base, synergy_value, n_terms_active)
    """
    if alpha is None:
        alpha = {d: 1/6 for d in DIMS}
    if synergy is None:
        synergy = SYNERGY_V7
    
    # CD基座: Π dim_i^α_i
    cd_base = 1.0
    for d in DIMS:
        v = max(dim_scores.get(d, 0), 0.1)
        cd_base *= v ** alpha.get(d, 1/6)
    
    # 交互项: Σ s_ij × R_ij
    synergy_total = 0.0
    active_terms = 0
    for (d1, d2), s in synergy.items():
        v1 = dim_scores.get(d1, 0)
        v2 = dim_scores.get(d2, 0)
        if v1 > 0 and v2 > 0:
            r_val = R_normalized(v1, v2)
            synergy_total += s * r_val
            if r_val > 0.01:
                active_terms += 1
    
    # 综合
    aic = A * cd_base * (1 + eta * synergy_total)
    return aic, cd_base, synergy_total, active_terms


def load_data():
    """加载dim_scores.json, 返回排序后的期列表"""
    data = json.loads(DIMS_FILE.read_text(encoding="utf-8"))
    periods_raw = data["periods"]
    keys = sorted(periods_raw.keys())
    
    periods = []
    for key in keys:
        raw = periods_raw[key]
        if isinstance(raw, dict) and "scores" in raw:
            dims = {k: v for k, v in raw["scores"].items() if k in DIMS}
        else:
            dims = {k: v for k, v in raw.items() if k in DIMS}
        note = raw.get("note", "") if isinstance(raw, dict) else ""
        periods.append({"date": key, "scores": dims, "note": note})
    
    return periods


# ═══════════════════════════════════════════════════════════
#  Step 1: 验证R函数性质
# ═══════════════════════════════════════════════════════════

def validate_R_function():
    """验证R函数是否满足:
    1. R ∈ [0, 0.5]
    2. R(λv_i, λv_j) = R(v_i, v_j)  (0次齐次)
    3. 基线时R≈0, 当前时R在0.3~0.5
    """
    print("=" * 75)
    print("Step 1: 验证R函数性质")
    print("=" * 75)
    
    # 1. 量纲非爆炸性
    print("\n1.1 量纲验证 (R是否随量级爆炸):")
    test_cases = [
        ("(5, 5)", 5, 5),
        ("(10, 10)", 10, 10),
        ("(100, 100)", 100, 100),
        ("(1000, 1000)", 1000, 1000),
        ("(3000, 3000)", 3000, 3000),
        ("(5, 100)", 5, 100),
        ("(1000, 50)", 1000, 50),
        ("(2750, 1580)", 2750, 1580),
    ]
    for label, v1, v2 in test_cases:
        r = R_normalized(v1, v2)
        print(f"  R{label} = {r:.4f}  {'(合规)' if r <= 0.5 else '(异常!)'}")
    
    # 2. 0次齐次验证
    print("\n1.2 0次齐次验证 (R(λv) == R(v)):")
    v1, v2 = 100, 200
    for lam in [0.1, 0.5, 2, 10, 100]:
        r1 = R_normalized(v1, v2)
        r2 = R_normalized(v1 * lam, v2 * lam)
        diff = abs(r1 - r2)
        status = "[OK]" if diff < 1e-10 else "[FAIL]"
        print(f"  λ={lam:>5}: R({v1},{v2}) = {r1:.6f} → R({v1*lam:.0f},{v2*lam:.0f}) = {r2:.6f}  {status}")
    
    # 3. 全历史R值范围
    periods = load_data()
    print("\n1.3 全历史R值范围:")
    all_r = {pair: [] for pair in SYNERGY_V7}
    for p in periods:
        for (d1, d2) in SYNERGY_V7:
            v1 = p["scores"].get(d1, 0)
            v2 = p["scores"].get(d2, 0)
            all_r[(d1, d2)].append(R_normalized(v1, v2))
    
    for (d1, d2), vals in all_r.items():
        min_r, max_r = min(vals), max(vals)
        last_r = vals[-1]
        print(f"  {d1}×{d2}: R∈[{min_r:.3f}, {max_r:.3f}]  当前R={last_r:.3f}")
    
    # 基线 vs 当前
    base_scores = periods[0]["scores"]
    curr_scores = periods[-1]["scores"]
    print(f"\n  R总和 (基线{periods[0]['date']}): {sum(R_normalized(base_scores.get(d1,0), base_scores.get(d2,0)) for (d1,d2) in SYNERGY_V7):.3f}")
    print(f"  R总和 (当前{periods[-1]['date']}): {sum(R_normalized(curr_scores.get(d1,0), curr_scores.get(d2,0)) for (d1,d2) in SYNERGY_V7):.3f}")
    
    return all_r


# ═══════════════════════════════════════════════════════════
#  Step 2: 纯几何平均试跑 (η=0)
# ═══════════════════════════════════════════════════════════

def run_geometric_mean(eta=0.0):
    """等权α=1/6, 纯CD或CD+交互, 不校准A"""
    periods = load_data()
    
    results = []
    for p in periods:
        dims = p["scores"]
        aic_raw, cd, syn, n_terms = calc_aic_v7(dims, eta=eta, A=1.0)
        results.append({"date": p["date"], "aic_raw": aic_raw, "cd": cd, 
                       "synergy": syn, "note": p["note"]})
    
    base_aic = results[0]["aic_raw"]
    
    # 归一化使基线=1000
    A_cal = 1000 / base_aic if base_aic > 0 else 1.0
    for r in results:
        r["aic"] = r["aic_raw"] * A_cal
    
    curr = results[-1]
    base = results[0]
    growth = curr["aic"] / base["aic"]
    
    return results, growth, A_cal


def print_geometric_results(label, results, growth, A_cal, eta):
    """打印几何平均结果"""
    print(f"\n{'=' * 75}")
    print(f"{label}")
    print(f"{'=' * 75}")
    print(f"  校准常数A={A_cal:.4f}")
    print(f"  η(交互强度)={eta}")
    print(f"  基线({results[0]['date']}): AIC={results[0]['aic']:.1f}  (cd={results[0]['cd']:.1f}, syn={results[0]['synergy']:.4f})")
    print(f"  当前({results[-1]['date']}): AIC={results[-1]['aic']:.1f}  (cd={results[-1]['cd']:.1f}, syn={results[-1]['synergy']:.4f})")
    print(f"  增长倍数: ×{growth:.1f}")
    
    # 关键基线
    key_dates = ["2022-12-01", "2024-01-01", "2025-01-01", "2026-01-01", "2026-06-16"]
    date_map = {r["date"]: r for r in results}
    
    print(f"\n{'日期':<14} {'AIC':>8} {'CD基座':>8} {'交互值':>8} {'倍数':>6}")
    print("-" * 46)
    for d in key_dates:
        if d in date_map:
            r = date_map[d]
            mult = r["aic"] / results[0]["aic"]
            print(f"{d:<14} {r['aic']:>8.1f} {r['cd']:>8.1f} {r['synergy']:>8.4f} {mult:>6.1f}x")
    
    return growth


# ═══════════════════════════════════════════════════════════
#  Step 3: 调η — 寻找合理增长倍数区间
# ═══════════════════════════════════════════════════════════

def tune_eta(eta_range=None):
    """扫描不同η值下的增长倍数"""
    if eta_range is None:
        eta_range = [0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    periods = load_data()
    
    print(f"\n{'=' * 75}")
    print("Step 3: η调校 — 寻找合理增长倍数区间")
    print(f"{'=' * 75}")
    print(f"\n{'η':>6} {'基线AIC':>10} {'当前AIC':>10} {'增长倍数':>10} {'交互占比':>10} {'交互项和':>10}")
    print("-" * 58)
    
    for eta in eta_range:
        # 第一次扫描先不归一化, 获取原始数值
        base_raw, _, _, _ = calc_aic_v7(periods[0]["scores"], eta=eta, A=1.0)
        curr_raw, _, curr_syn, _ = calc_aic_v7(periods[-1]["scores"], eta=eta, A=1.0)
        
        # 归一化
        A_cal = 1000 / base_raw if base_raw > 0 else 1.0
        base_aic = base_raw * A_cal
        curr_aic = curr_raw * A_cal
        growth = curr_aic / base_aic
        
        # 交互占比 = (CD* (1+ηΣ) - CD) / (CD*(1+ηΣ)) = ηΣ/(1+ηΣ)
        # 交互占比 = η×syn_total / (1+η×syn_total)
        _, _, curr_syn_t, _ = calc_aic_v7(periods[-1]["scores"], eta=eta, A=1.0)
        
        # 交互对AIC的实际贡献百分比
        # AIC = A × CD × (1+η×syn_total)
        # 交互贡献 = (1+η×syn_total) - 1 = η×syn_total
        # 占比 = η×syn / (1+η×syn)
        syn_contrib_pct = eta * curr_syn_t / (1 + eta * curr_syn_t) * 100 if eta > 0 else 0
        
        print(f"{eta:>6.1f} {base_aic:>10.1f} {curr_aic:>10.1f} {growth:>10.1f}x {syn_contrib_pct:>9.1f}% {curr_syn_t:>10.4f}")
    
    print(f"\n  目标增长倍数: 30~80x (用户指定)")
    print(f"  推荐η区间: 0~1.0 (交互占比<30%)")


# ═══════════════════════════════════════════════════════════
#  Step 4: 完整AIC时间序列 + 对比v3
# ═══════════════════════════════════════════════════════════

def build_timeseries_v7(alpha=None, eta=1.0, synergy=None):
    """构建v7完整时间序列"""
    if alpha is None:
        alpha = {d: 1/6 for d in DIMS}
    if synergy is None:
        synergy = SYNERGY_V7
    
    periods = load_data()
    
    # 首次扫描确定A
    results_raw = []
    for p in periods:
        aic_raw, cd, syn, n_terms = calc_aic_v7(p["scores"], alpha=alpha, eta=eta, 
                                                synergy=synergy, A=1.0)
        results_raw.append({"date": p["date"], "aic_raw": aic_raw, "cd": cd, 
                           "synergy": syn, "n_terms": n_terms, "note": p["note"]})
    
    A_cal = 1000 / results_raw[0]["aic_raw"] if results_raw[0]["aic_raw"] > 0 else 1.0
    
    results = []
    prev_aic = None
    for r in results_raw:
        aic = r["aic_raw"] * A_cal
        aidi = aic - prev_aic if prev_aic is not None else 100.0
        results.append({**r, "aic": round(aic, 1), "aidi": round(aidi, 1)})
        prev_aic = aic
    
    return results, A_cal


def compare_v3_vs_v7():
    """v3 vs v7全量对比"""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    
    print("\n" + "=" * 75)
    print("Step 4: v3(当前) vs v7(新公式) 全量对比")
    print("=" * 75)
    
    # v3数据
    from engine.aidicore_v3 import build_timeseries_v3
    v3_periods = build_timeseries_v3(normalize=True)
    v3_map = {p["date"]: p for p in v3_periods}
    
    # 测试不同η
    for eta_test in [0, 0.5, 1.0, 2.0]:
        v7_results, A_cal = build_timeseries_v7(eta=eta_test)
        v7_map = {r["date"]: r for r in v7_results}
        
        base_v3 = v3_periods[0]
        curr_v3 = v3_periods[-1]
        base_v7 = v7_results[0]
        curr_v7 = v7_results[-1]
        
        print(f"\n── η={eta_test:.1f} ──")
        print(f"{'指标':<20} {'v3(当前)':>12} {'v7(新)':>12} {'差异':>10}")
        print("-" * 56)
        print(f"{'基线AIC':<20} {base_v3['aic']:>12,} {base_v7['aic']:>12.1f} {'—':>10}")
        print(f"{'当前AIC':<20} {curr_v3['aic']:>12,} {curr_v7['aic']:>12.1f} "
              f"{(curr_v7['aic']-curr_v3['aic'])/curr_v3['aic']*100:>+9.1f}%")
        print(f"{'增长倍数':<20} ×{curr_v3['aic']/base_v3['aic']:>8.1f} ×{curr_v7['aic']/base_v7['aic']:>8.1f}")
        
        # 关键基线对比
        key_dates = ["2022-12-01", "2024-01-01", "2025-01-01", "2026-01-01", "2026-06-16"]
        print(f"\n{'日期':<14} {'v3 AIC':>10} {'v7 AIC':>10} {'v7倍率':>8} {'v3倍率':>8}")
        print("-" * 52)
        for d in key_dates:
            v3p = v3_map.get(d)
            v7p = v7_map.get(d)
            if v3p and v7p:
                print(f"{d:<14} {v3p['aic']:>10,} {v7p['aic']:>10.1f} "
                      f"{v7p['aic']/v7_results[0]['aic']:>8.1f}x {v3p['aic']/v3_periods[0]['aic']:>8.1f}x")


# ═══════════════════════════════════════════════════════════
#  主运行
# ═══════════════════════════════════════════════════════════

def run_full_v7():
    """完整运行所有验证步骤"""
    
    # ── Step 1: R函数验证 ──
    validate_R_function()
    
    # ── Step 2: 纯几何平均 ──
    print("\n\n" + "█" * 75)
    print("█ Step 2: 纯几何平均试跑 (η=0)")
    print("█" * 75)
    results, growth, A_cal = run_geometric_mean(eta=0.0)
    print_geometric_results("纯几何平均 (α=1/6, η=0, 无交互)", results, growth, A_cal, 0)
    
    # ── Step 3: 调η ──
    tune_eta()
    
    # ── Step 4: 推荐η=1.0完整结果 ──
    print("\n\n" + "█" * 75)
    print("█ Step 3+: 推荐配置 (α=1/6, η=1.0)")
    print("█" * 75)
    results_v7_1, A_cal_1 = build_timeseries_v7(eta=1.0)
    growth_1 = results_v7_1[-1]["aic"] / results_v7_1[0]["aic"]
    print_geometric_results("推荐CD+交互 (α=1/6, η=1.0)", results_v7_1, growth_1, A_cal_1, 1.0)
    
    # ── 对比v3 ──
    compare_v3_vs_v7()
    
    # ── 保存报告 ──
    report = {
        "meta": {
            "engine": "AIDI v7 Cobb-Douglas AIC",
            "generated": "2026-06-22",
            "formula": "AIC = A × Πdim^α × (1 + η×Σs_ij×R_ij)",
            "R_function": "R(v_i,v_j) = min²/(v_i²+v_j²)  (0次齐次)",
            "recommended_eta": 1.0,
        },
        "validation": {
            "R_is_homogeneous_order_0": True,
            "R_range": "[0, 0.5]",
            "geo_mean_growth_x": round(results[0]["aic"] / 1000, 1),
        },
        "eta_sweep": {
            "pure_geo_mean": {"eta": 0, "growth_x": round(growth, 1), "aic_current": round(results[-1]["aic"], 1)},
            "cd_plus_interaction": {"eta": 1.0, "growth_x": round(growth_1, 1), "aic_current": round(results_v7_1[-1]["aic"], 1)},
        },
        "key_periods_v7_recommended": [
            {"date": r["date"], "aic": r["aic"], "cd_base": round(r["cd"], 1),
             "synergy": round(r["synergy"], 4), "growth_x": round(r["aic"] / results_v7_1[0]["aic"], 1)}
            for r in results_v7_1
            if r["date"] in ["2022-12-01", "2024-01-01", "2025-01-01", "2026-01-01", "2026-06-16"]
        ],
    }
    
    out = BASE_DIR / "reports/aidi_v7_design_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告保存: {out}")


if __name__ == "__main__":
    run_full_v7()
