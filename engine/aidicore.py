#!/usr/bin/env python3
"""
AIDI Engine v1 — AI Development Index 核心引擎
=============================================
功能: 六维评分 → 加权AIDI → 贝叶斯预测 → 校验修正
数学基础: 数理科学助手 (贝叶斯推断/结构化验证)
算法引擎: Hermes (PSO/GA权重优化)
"""
import json
import math
import sys
import statistics
from datetime import datetime
from pathlib import Path

# ── 默认配置 ──────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "model": 0.25,      # 模型能力 — 推理/生成质量
    "hardware": 0.20,   # 硬件算力 — 训练/推理速度
    "algorithm": 0.20,  # 算法创新 — 效率/成本下降
    "business": 0.15,   # 商业生态 — 价格/融资/创业
    "adoption": 0.10,   # 应用渗透 — 实际落地/用户
    "opensource": 0.10, # 开源生态 — GitHub/社区
}
DIMS = list(DEFAULT_WEIGHTS.keys())


def load_history(path="data/aidi_history.json"):
    """加载AIDI历史数据"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def calc_aidi(dimensions, weights=None):
    """计算AIDI指数: 加权六维得分"""
    weights = weights or DEFAULT_WEIGHTS
    return round(sum(dimensions[d] * weights[d] for d in DIMS))


def calc_mape(actual, predicted):
    """平均绝对百分比误差 (MAPE)"""
    if actual == 0: return 0
    return abs(actual - predicted) / actual


def bayesian_predict(history, weights=None):
    """贝叶斯预测: 基于历史趋势+波动性预测下期AIDI"""
    # 提取最近几期的AIDI值和变化量
    aidis = [h["aidi"] for h in history[-6:]]  # 最近6期
    
    if len(aidis) < 2:
        return {"predicted": None, "ci_low": None, "ci_high": None}
    
    # 计算变化量
    deltas = [aidis[i] - aidis[i-1] for i in range(1, len(aidis))]
    
    # 加权平均变化量 (最近的变化权重更大)
    weights_ts = [0.1, 0.15, 0.2, 0.25, 0.3][:len(deltas)]
    w_sum = sum(weights_ts)
    weights_ts = [w/w_sum for w in weights_ts]
    avg_delta = sum(d * w for d, w in zip(deltas, weights_ts))
    
    # 波动性 (标准差)
    if len(deltas) > 1:
        variance = sum((d - avg_delta)**2 for d in deltas) / len(deltas)
        volatility = math.sqrt(variance)
    else:
        volatility = abs(avg_delta) * 0.5 if avg_delta else 10
    
    # 预测 = 最新值 + 加权趋势
    latest = aidis[-1]
    predicted = round(latest + avg_delta)
    
    # 置信区间 (95%): ±1.96 × 波动性
    ci = round(1.96 * volatility)
    
    return {
        "predicted": predicted,
        "ci_low": predicted - ci,
        "ci_high": predicted + ci,
        "confidence": round(max(0, min(100, 100 - ci / predicted * 100))),
        "trend_delta": round(avg_delta, 1),
        "volatility": round(volatility, 1)
    }


def calc_validation(history, prediction_results=None):
    """校验函数: 评估历史预测精度，计算校验值 F"""
    if len(history) < 3:
        return {"F": None, "note": "数据不足"}
    
    # 模拟"回头看"校验: 对每一期用之前的数据做预测，对比实际值
    errors = []
    for i in range(2, len(history)):
        # 用前i期数据预测第i+1期
        pred = bayesian_predict(history[:i])
        if pred["predicted"]:
            actual = history[i]["aidi"]
            errors.append({
                "date": history[i]["date"],
                "actual": actual,
                "predicted": pred["predicted"],
                "ci_low": pred["ci_low"],
                "ci_high": pred["ci_high"],
                "error": actual - pred["predicted"],
                "in_ci": pred["ci_low"] <= actual <= pred["ci_high"]
            })
    
    if not errors:
        return {"F": None}
    
    # 计算校验值 F
    mape = sum(abs(e["error"]) for e in errors) / sum(e["actual"] for e in errors)
    in_ci_rate = sum(1 for e in errors if e["in_ci"]) / len(errors)
    calibration = 1 - abs(in_ci_rate - 0.95)  # 越接近95%越好
    F = round((1 - mape) * 0.6 + calibration * 0.4, 4)
    
    return {
        "F": F,
        "mape": round(mape, 4),
        "in_ci_rate": round(in_ci_rate, 2),
        "calibration": round(calibration, 4),
        "total_checks": len(errors),
        "errors": errors[-5:]  # 最近5次
    }


def optimize_weights(history, method="pso"):
    """用PSO/GA思想优化权重 (简化版: 网格搜索)"""
    from itertools import product
    
    best = {"weights": DEFAULT_WEIGHTS.copy(), "mape": 1.0}
    
    # 五档权重搜索 (粗调)
    steps = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    
    # 限制: 主维度(模型/硬件/算法) 0.15-0.35, 次维度 0.05-0.25
    candidates = []
    for m in [0.20, 0.25, 0.30]:
        for h in [0.15, 0.20, 0.25]:
            for a in [0.15, 0.20, 0.25]:
                for b in [0.10, 0.15]:
                    for ad in [0.05, 0.10]:
                        for op in [0.05, 0.10]:
                            total = m + h + a + b + ad + op
                            if abs(total - 1.0) < 0.01:
                                candidates.append({
                                    "model": m, "hardware": h, "algorithm": a,
                                    "business": b, "adoption": ad, "opensource": op
                                })
    
    for w in candidates:
        errors = []
        for i in range(1, len(history)):
            dims = history[i]["dimensions"]
            predicted = calc_aidi(dims, w)
            errors.append(abs(predicted - history[i]["aidi"]) / max(history[i]["aidi"], 1))
        
        mape = sum(errors) / len(errors)
        if mape < best["mape"]:
            best = {"weights": w.copy(), "mape": round(mape, 4)}
    
    return best


def compute_dimension_scores(fresh_data):
    """
    根据最新采集数据计算六维评分 (0-1000)
    由千寻搜索 + 数据分析师完成
    """
    # 这是一个接口占位 — 实际数据由采集步骤提供
    return fresh_data


# ════════════════════════════════════════════════════════
# AIC 能力曲线 + 基准对齐
# ════════════════════════════════════════════════════════

def calc_aic(history):
    """
    AIC (AI Capability Index) = 能力总量
    AIC(t) = AIC(t-1) + (AIDI(t) + AIDI(t-1)) / 2 * dt
    即: AIDI的积分 = 发展速度的累计 = 能力总量

    物理意义:
      AIDI = 速度 (发展有多快)
      AIC  = 位移 (已经走了多远)
    """
    aics = []
    baseline_aic = 1000  # 基线: 2026-01-01 能力=1000

    for i, h in enumerate(history):
        if i == 0:
            aic = baseline_aic
        else:
            # 梯形积分: (v_prev + v_curr) / 2 * dt
            # dt = 0.5个月 (每半月一期)
            dt = 0.5
            avg_velocity = (history[i-1]["aidi"] + h["aidi"]) / 2
            aic = aics[-1] + avg_velocity * dt / 100

        aics.append(round(aic, 1))

    return aics


def load_benchmarks(path="data/benchmarks.json"):
    """加载外部基准测评数据"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"records": []}


def calibrate_aic(history, benchmarks):
    """
    AIC与外部基准测评对齐
    计算: 模型算的AIC vs 实际测评结论 → 偏差 → 调整因子
    """
    aics = calc_aic(history)

    # 给历史数据附上AIC
    for i, h in enumerate(history):
        h["aic"] = aics[i]

    if not benchmarks.get("records"):
        return {
            "status": "no_benchmarks",
            "aics": aics,
            "note": "无基准数据, AIC为纯模型推算"
        }

    # 对比每期AIC与外部基准
    calibrations = []
    adj_factors = []

    for bm in benchmarks["records"]:
        # 找对应日期的AIC
        match = [h for h in history if h["date"] == bm["date"]]
        if not match:
            continue

        h = match[0]
        aic_model = h["aic"]
        aic_actual = bm["aic_actual"]  # 外部测评给出的实际能力值

        # 偏差
        gap = aic_actual - aic_model
        gap_pct = gap / aic_model if aic_model else 0

        calibrations.append({
            "date": bm["date"],
            "aic_model": aic_model,
            "aic_actual": aic_actual,
            "gap": round(gap, 1),
            "gap_pct": round(gap_pct, 4),
            "benchmark_source": bm.get("source", ""),
            "benchmark_event": bm.get("event", ""),
        })

        # 调整因子 = 实际/模型
        if aic_model > 0:
            adj_factors.append(aic_actual / aic_model)

    # 综合调整因子 (加权滑动平均)
    if adj_factors:
        recent = adj_factors[-3:] if len(adj_factors) >= 3 else adj_factors
        # 最近几次的权重更大
        w = [1/len(recent)] * len(recent)
        adjustment_factor = sum(a * w_i for a, w_i in zip(recent, w))
    else:
        adjustment_factor = 1.0

    return {
        "status": "calibrated",
        "aics": aics,
        "adjustment_factor": round(adjustment_factor, 4),
        "calibrations": calibrations,
        "total_calibrations": len(calibrations),
        "note": "调整因子 = 外部基准实测 / 模型推算AIC"
    }


def analyze_calibration_gap(calibrations):
    """
    分析AIC偏差的原因: 为什么模型算的跟实际测评不一样?
    积累经验 → 作为今后的调整因子
    """
    if not calibrations:
        return {"经验": "暂无校准数据", "调整建议": "默认因子1.0"}

    gaps = [c["gap_pct"] for c in calibrations]
    avg_gap = statistics.mean(gaps) if gaps else 0

    # 偏差模式分析
    pattern = ""
    if abs(avg_gap) < 0.02:
        pattern = "高精度 — 模型AIC与外部基准基本吻合"
    elif avg_gap > 0:
        pattern = "系统偏高 — 模型高估了实际能力, 建议下调"
    else:
        pattern = "系统偏低 — 模型低估了实际能力(新模型发布密集期常见), 建议上调"

    # 偏差原因分类
    reasons = []
    for c in calibrations:
        if abs(c["gap_pct"]) > 0.05:
            reason = {
                "date": c["date"],
                "gap_pct": c["gap_pct"],
                "推测原因": "",
                "benchmark_event": c["benchmark_event"]
            }
            if c["gap_pct"] > 0.05:
                reason["推测原因"] = "模型维度得分过高估算了实际能力提升, 可能因为商业宣传夸大了技术进展"
            else:
                reason["推测原因"] = "外部基准反映了实际突破性进展, 但模型维度得分未能及时捕捉"
            reasons.append(reason)

    return {
        "avg_gap_pct": round(avg_gap, 4),
        "pattern": pattern,
        "偏差分析": reasons,
        "调整建议": f"建议下期使用调整因子 {round(1 + avg_gap, 4)} 修正AIC"
    }


def full_run(history_path="data/aidi_history.json", fresh_dimensions=None, benchmarks_path="data/benchmarks.json"):
    """全流程运行 (含AIC能力曲线+基准对齐)"""
    # 1. 加载历史
    data = load_history(history_path)
    history = data["history"]
    meta = data["meta"]
    
    # 2. 权重优化
    opt = optimize_weights(history)
    print(f"[权重优化] MAPE={opt['mape']}, 权重={opt['weights']}")
    
    # 3. 如果有新数据, 计算最新AIDI
    if fresh_dimensions:
        new_aidi = calc_aidi(fresh_dimensions, opt["weights"])
        print(f"[AIDI] 最新: {new_aidi}")
    else:
        new_aidi = history[-1]["aidi"]
    
    # 4. AIC能力曲线
    aics = calc_aic(history)
    for i, h in enumerate(history):
        h["aic"] = aics[i]
    
    current_aic = aics[-1]
    print(f"[AIC] 当前能力: {current_aic}, 较基线 +{round(current_aic - 1000, 1)}")
    
    # 5. 与外部基准对齐
    benchmarks = load_benchmarks(benchmarks_path)
    calib = calibrate_aic(history, benchmarks)
    print(f"[基准对齐] {calib['status']}, 调整因子={calib.get('adjustment_factor', 'N/A')}")
    
    if calib["status"] == "calibrated" and calib["calibrations"]:
        gap_analysis = analyze_calibration_gap(calib["calibrations"])
        print(f"[偏差分析] {gap_analysis['pattern']}")
        print(f"[调整建议] {gap_analysis['调整建议']}")
    else:
        gap_analysis = {"经验": "暂无校准数据"}
    
    # 6. 贝叶斯预测
    pred = bayesian_predict(history)
    print(f"[预测] 下期AIDI: {pred['predicted']} (CI: {pred['ci_low']}~{pred['ci_high']})")
    
    # 7. 校验
    val = calc_validation(history)
    if val["F"]:
        print(f"[校验] F={val['F']}, MAPE={val['mape']}, 区间命中率={val['in_ci_rate']}")
    
    # 8. 输出报告
    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "current_aidi": new_aidi,
        "current_aic": current_aic,
        "aic_baseline": 1000,
        "aic_gain": round(current_aic - 1000, 1),
        "current_dimensions": fresh_dimensions,
        "weights": opt["weights"],
        "aic_calibration": {
            "adjustment_factor": calib.get("adjustment_factor"),
            "calibrations": calib.get("calibrations", []),
            "gap_analysis": gap_analysis
        },
        "prediction": pred,
        "validation": val,
        "history_length": len(history)
    }
    
    return report


if __name__ == "__main__":
    report = full_run()
    print("\n" + "="*50)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    
    # 保存报告
    report_path = f"reports/aidi_report_{datetime.now().strftime('%Y%m%d')}.json"
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n报告已保存: {report_path}")
