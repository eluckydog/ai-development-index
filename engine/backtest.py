"""
AIDI Backtest — v5 AIC预测 + v6 能力演化回测
================================================
检验引擎在历史截断点上的预测精度与因果稳定性.

截断点: 2023-06, 2023-12, 2024-06, 2025-01, 2025-06
预测目标: AIC在+6/+12/+18个月后的实际值
"""
import json, sys, math
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent.parent

# 加载v5预测器
sys.path.insert(0, str(BASE_DIR))
from engine.aidicore_v5 import AIDIPredictor
from engine.aidicore_v6 import CapabilityEvolutionEngine, granger_causality, cross_correlation

# ── 加载完整数据 ──────────────────────────────────────────
DIMS_FILE = BASE_DIR / "data/curated/dim_scores.json"
data = json.loads(DIMS_FILE.read_text(encoding="utf-8"))
periods_raw = data["periods"]

DIMS = ["intelligence", "multimodal", "agent", "programming", "knowledge", "ecosystem"]

from engine.aidicore_v5 import _calc_aic, SYNERGY

# 构建完整AIC序列
keys = sorted(periods_raw.keys())
full_aic = {}
prev_aic, base_raw = None, None
for key in keys:
    raw = periods_raw[key]
    dims = {k: v for k, v in raw["scores"].items() if k in DIMS} \
           if isinstance(raw, dict) and "scores" in raw \
           else {k: v for k, v in raw.items() if k in DIMS}
    aic_raw = _calc_aic(dims)
    if base_raw is None: base_raw = aic_raw
    aic = round(aic_raw / base_raw * 1000)
    full_aic[key] = {"aic": aic, "aid": aic - prev_aic if prev_aic is not None else 100}
    prev_aic = aic

aic_series_full = np.array([v["aic"] for v in full_aic.values()])
date_keys = list(full_aic.keys())
total_periods = len(date_keys)


# ═══════════════════════════════════════════════════════════
#  回测1: v5 AIC预测精度
# ═══════════════════════════════════════════════════════════

#  截断点设定: 日期 -> 该日期在序列中的索引
CUTOFFS = {
    "2023-06-01": {"idx": None, "note": "ChatGPT半年, GPT-4刚发布"},
    "2023-12-01": {"idx": None, "note": "GPT-4 Turbo时代"},
    "2024-06-01": {"idx": None, "note": "GPT-4o发布月"},
    "2025-01-01": {"idx": None, "note": "o3+DeepSeek R1"},
    "2025-06-01": {"idx": None, "note": "GPT-5前夕"},
}

for cutoff in CUTOFFS:
    for i, d in enumerate(date_keys):
        if d >= cutoff:
            CUTOFFS[cutoff]["idx"] = i
            break
    if CUTOFFS[cutoff]["idx"] is None:
        CUTOFFS[cutoff]["idx"] = total_periods - 1

# 预测目标: +6期(3个月), +12期(6个月), +18期(9个月), +24期(12个月)
HORIZONS = [6, 12, 18, 24]

print("=" * 90)
print("AIDI Backtest: v5 AIC 预测精度回测")
print("=" * 90)

results_v5 = []

for cutoff_name, cutoff_info in CUTOFFS.items():
    cutoff_idx = cutoff_info["idx"]
    cutoff_note = cutoff_info["note"]
    
    # 截断数据
    aic_truncated = aic_series_full[:cutoff_idx + 1]
    dates_truncated = date_keys[:cutoff_idx + 1]
    current_aic = aic_truncated[-1]
    
    print(f"\n{'─' * 90}")
    print(f"截断点: {cutoff_name} (idx={cutoff_idx}/{total_periods}) | {cutoff_note}")
    print(f"  当时AIC: {int(current_aic):,}")
    print(f"{'─' * 90}")
    
    for h in HORIZONS:
        target_idx = cutoff_idx + h
        if target_idx >= total_periods:
            continue
        
        target_date = date_keys[target_idx]
        actual = aic_series_full[target_idx]
        
        # 用v5的预测AIC方法:
        # 重新构建预测器, 只给截断后的数据
        # 由于AIDIPredictor从文件读全量, 我们用函数直接算
        from engine.aidicore_v5 import holt_winters, arima_predict, exp_predict, bootstrap_ci
        
        hw_pred = holt_winters(aic_truncated, h)[0]
        arima_pred = arima_predict(aic_truncated, h=h)[0]
        # 指数预测: 使用截断数据的实际长度, 不超过24
        exp_recent_n = min(24, len(aic_truncated))
        exp_pred = exp_predict(aic_truncated, h, recent_n=exp_recent_n)
        hw_low, hw_high = bootstrap_ci(aic_truncated, h, n_bootstrap=500)
        
        aic_hw = hw_pred[-1]
        aic_arima = arima_pred[-1]
        aic_exp = exp_pred[-1]
        
        w_hw, w_arima, w_exp = 0.4, 0.3, 0.3
        agg = w_hw * aic_hw + w_arima * aic_arima + w_exp * aic_exp
        
        ci_low = w_hw * hw_low[-1] + w_arima * (aic_arima * 0.95) + w_exp * aic_exp * 0.85
        ci_high = w_hw * hw_high[-1] + w_arima * (aic_arima * 1.05) + w_exp * aic_exp * 1.15
        
        # 误差计算
        err_hw = (aic_hw - actual) / actual * 100
        err_arima = (aic_arima - actual) / actual * 100
        err_exp = (aic_exp - actual) / actual * 100
        err_agg = (agg - actual) / actual * 100
        
        # 区间覆盖
        in_ci = 1 if ci_low <= actual <= ci_high else 0
        
        is_best = "<- 最准" if min(abs(err_hw), abs(err_arima), abs(err_exp), abs(err_agg)) == abs(err_agg) else ""
        
        print(f"  +{h:>2}期({h//2}月) -> {target_date}")
        print(f"    实际AIC: {actual:>8,}")
        print(f"    Holt-W:  {aic_hw:>8,}  (偏差{err_hw:>+.1f}%)")
        print(f"    ARIMA:   {aic_arima:>8,}  (偏差{err_arima:>+.1f}%)")
        print(f"    指数:    {aic_exp:>8,}  (偏差{err_exp:>+.1f}%)")
        print(f"    聚合:    {agg:>8,}  (偏差{err_agg:>+.1f}%)  CI: [{int(ci_low):,}~{int(ci_high):,}] {'[OK]' if in_ci else '[XX]'}{is_best}")
        
        results_v5.append({
            "cutoff": cutoff_name,
            "horizon": h,
            "target_date": target_date,
            "actual": int(actual),
            "holtwinters": round(aic_hw),
            "arima": round(aic_arima),
            "exponential": round(aic_exp),
            "aggregate": round(agg),
            "err_agg_pct": round(err_agg, 2),
            "in_ci": in_ci,
        })

# ── 汇总统计 ──
print(f"\n\n{'=' * 90}")
print("v5 回测统计汇总")
print("=" * 90)

by_horizon = {}
for r in results_v5:
    h = r["horizon"]
    if h not in by_horizon: by_horizon[h] = []
    by_horizon[h].append(r)

print(f"\n{'预测期':>8} {'次数':>5} {'平均偏差':>10} {'|平均偏差|':>10} {'中位偏差':>10} {'区间覆盖率':>10}")
print("-" * 55)
overall_errs = []
for h in sorted(by_horizon.keys()):
    rows = by_horizon[h]
    errs = [r["err_agg_pct"] for r in rows]
    abs_errs = [abs(e) for e in errs]
    coverage = sum(r["in_ci"] for r in rows) / len(rows) * 100
    overall_errs.extend(errs)
    print(f"+{h:>2}期({h//2}月): {len(rows):>5}  {np.mean(errs):>+9.1f}%  {np.mean(abs_errs):>9.1f}%  {np.median(errs):>+9.1f}%  {coverage:>9.0f}%")

# 按截断点统计
print(f"\n按截断点:")
for cutoff in CUTOFFS:
    rows = [r for r in results_v5 if r["cutoff"] == cutoff]
    if not rows: continue
    errs = [r["err_agg_pct"] for r in rows]
    abs_errs = [abs(e) for e in errs]
    print(f"  {cutoff}: 平均偏差{np.mean(errs):>+.1f}%  |偏差|={np.mean(abs_errs):.1f}%")


# ═══════════════════════════════════════════════════════════
#  回测2: v6 Granger因果稳定性
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 90}")
print("AIDI Backtest: v6 Granger因果稳定性回测")
print("=" * 90)

# 获取v6的能力评分序列
evo = CapabilityEvolutionEngine()
v6_scores = evo.scores
v6_timeline = evo.timeline

# 各截断点在v6时间轴上的位置
v6_cutoffs = {
    "2023-06": {"timeline_idx": None},
    "2023-12": {"timeline_idx": None},
    "2024-06": {"timeline_idx": None},
    "2025-01": {"timeline_idx": None},
}

for c in v6_cutoffs:
    for i, d in enumerate(v6_timeline):
        if d >= c:
            v6_cutoffs[c]["timeline_idx"] = i
            break
    if v6_cutoffs[c]["timeline_idx"] is None:
        v6_cutoffs[c]["timeline_idx"] = len(v6_timeline) - 1

# 检验的因果链
CAUSAL_CHAINS = [
    ("algorithm_game", "text_llm", "算法游戏->文字"),
    ("text_llm", "image", "文字->图片"),
    ("image", "video", "图片->视频"),
    ("video", "world_model", "视频->世界模型"),
    ("world_model", "physical_interaction", "世界模型->物理交互"),
]

for cutoff_name, cutoff_info in v6_cutoffs.items():
    ci = cutoff_info["timeline_idx"]
    print(f"\n截断点: {cutoff_name}")
    print(f"{'因果链':<30} {'成立?':>6} {'p值':>10} {'效应量':>8}")
    print("-" * 56)
    
    for a, b, label in CAUSAL_CHAINS:
        if a not in v6_scores or b not in v6_scores:
            continue
        x = v6_scores[a][:ci+1]
        y = v6_scores[b][:ci+1]
        if len(x) < 6 or len(y) < 6:
            print(f"{label:<30} {'数据不足':>6}")
            continue
        
        result = granger_causality(x, y, max_lag=4)
        sig = "成立" if result["causal"] else "不成立"
        best_p = result["best_p_value"] if result["best_p_value"] else 0
        best_es = result["best_effect_size"] if result["best_effect_size"] else 0
        print(f"{label:<30} {sig:>6} {best_p:>10.4f} {best_es:>8.4f}")

# 稳定性汇总: 当前完整数据的因果结论
print(f"\n\n因果稳定性总结:")
print(f"{'因果链':<30} {'完整数据':>8} {'最早成立':>10} {'稳定性':>8}")
print("-" * 58)

for a, b, label in CAUSAL_CHAINS:
    if a not in v6_scores or b not in v6_scores:
        continue
    
    # 完整数据的因果结论
    x_full = v6_scores[a]
    y_full = v6_scores[b]
    full_result = granger_causality(x_full, y_full, max_lag=4)
    full_causal = full_result["causal"]
    
    # 最早成立的截断点
    first_established = None
    for c_name, c_info in v6_cutoffs.items():
        ci2 = c_info["timeline_idx"]
        if ci2 < 6: continue
        x_sub = v6_scores[a][:ci2+1]
        y_sub = v6_scores[b][:ci2+1]
        if len(x_sub) < 6 or len(y_sub) < 6: continue
        sub_result = granger_causality(x_sub, y_sub, max_lag=4)
        if sub_result["causal"]:
            first_established = c_name
            break
    
    stability = "稳定" if first_established is not None else "待验证"
    full_mark = "成立" if full_causal else "不成立"
    print(f"{label:<30} {full_mark:>8} {str(first_established or '—'):>10} {stability:>8}")


# ═══════════════════════════════════════════════════════════
#  回测3: v6 演化S曲线预测回测
# ═══════════════════════════════════════════════════════════

print(f"\n\n{'=' * 90}")
print("AIDI Backtest: v6 能力演化S曲线回测")
print("=" * 90)

# 对每个Phase 3能力, 看早期S曲线拟合是否预测对了后期发展
for cap_id in ["world_model", "spatial_reasoning", "physical_interaction"]:
    if cap_id not in v6_scores: continue
    
    series = v6_scores[cap_id]
    cap_name = {"world_model": "世界模型", "spatial_reasoning": "空间推理", 
                "physical_interaction": "物理交互"}[cap_id]
    
    print(f"\n{cap_name} ({cap_id}):")
    print(f"{'截断':>10} {'实际当前':>8} {'预测当前':>8} {'偏差':>8}")
    print("-" * 38)
    
    for cutoff_name, cutoff_info in v6_cutoffs.items():
        ci = cutoff_info["timeline_idx"]
        if ci < 4: continue
        
        # 用截断前的数据拟合S曲线
        y_sub = series[:ci+1]
        y_norm = y_sub / 1000.0
        t_sub = np.arange(len(y_norm), dtype=float)
        
        from engine.aidicore_v6 import fit_s_curve, logistic_model
        params = fit_s_curve(y_norm)
        if params is None: continue
        
        K, a, b, t0 = params
        
        # 预测到当前时间点
        future_t = np.arange(len(y_sub), len(series), dtype=float)
        if len(future_t) == 0: continue
        pred_now = logistic_model(future_t, K, a, b, t0) * 1000
        
        actual_now = series[-1]
        predicted_value = pred_now[-1]
        err = (predicted_value - actual_now) / actual_now * 100
        
        # 该截断点时的S曲线拟合质量
        fitted_past = logistic_model(t_sub, K, a, b, t0) * 1000
        ss_res = np.sum((y_sub - fitted_past)**2)
        ss_tot = np.sum((y_sub - np.mean(y_sub))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        
        print(f"{cutoff_name:>10} {actual_now:>8.0f} {predicted_value:>8.0f} {err:>+7.1f}%  (拟合R^2={r2:.3f})")


# ═══════════════════════════════════════════════════════════
#  保存回测报告
# ═══════════════════════════════════════════════════════════

# 先构建因果稳定性数据
causal_stability = []
for a, b, label in CAUSAL_CHAINS:
    if a not in v6_scores or b not in v6_scores:
        continue
    x_full = v6_scores[a]
    y_full = v6_scores[b]
    r_full = granger_causality(x_full, y_full, max_lag=4)
    first_est = None
    for cn in v6_cutoffs:
        ci = v6_cutoffs[cn]["timeline_idx"]
        if ci < 6: continue
        if len(x_full[:ci+1]) < 6 or len(y_full[:ci+1]) < 6: continue
        r_sub = granger_causality(x_full[:ci+1], y_full[:ci+1], max_lag=4)
        if r_sub["causal"]:
            first_est = cn
            break
    causal_stability.append({
        "pair": label,
        "full_data_causal": r_full["causal"],
        "first_established": first_est,
    })

backtest_result = {
    "meta": {
        "title": "AIDI v5+v6 双引擎回测报告",
        "generated": "2026-06-21",
        "cutoff_points": list(CUTOFFS.keys()),
        "v5_horizons": HORIZONS,
    },
    "v5_forecast_accuracy": {
        r["target_date"] + "@" + r["cutoff"]: {
            "cutoff": r["cutoff"],
            "horizon": r["horizon"],
            "target": r["target_date"],
            "actual": r["actual"],
            "predicted_aggregate": r["aggregate"],
            "error_pct": r["err_agg_pct"],
            "in_ci": r["in_ci"],
        } for r in results_v5
    },
    "v6_causal_stability": {
        "chains": causal_stability
    }
}

out = BASE_DIR / "reports/aidi_backtest_report.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(backtest_result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n\n回测报告保存: {out}")
print("=" * 90)
