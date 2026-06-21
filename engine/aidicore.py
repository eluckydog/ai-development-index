#!/usr/bin/env python3
"""
AIDI Engine v2 — AI Development Index 核心引擎 (修正版)
======================================================
核心逻辑:
  AIC (能力总量) = 基准测评映射 (0-1000)  ← 主指标
  AIDI (发展速度) = AIC的每半月增量         ← 派生指标

数学基础: 数理科学助手 (贝叶斯推断/结构化验证)
算法引擎: Hermes (PSO/GA权重优化)
"""
import json
import math
import sys
import statistics
from datetime import datetime
from pathlib import Path


# ── AIC基准映射表 ──────────────────────────────────────────────────
# 关键模型的外部基准分数 → AIC锚点
# AIC是0-1000分制, 1000 = 当前最强模型

AIC_ANCHORS = [
    # 日期, 事件, AIC (基线1000=GPT-4), 来源
    ("2022-12-01", "GPT-4发布", 1000, "OpenAI, MMLU 86.4%"),
    ("2023-03-15", "GPT-4正式上线", 1400, "OpenAI, 多模态+编程"),
    ("2023-07-15", "Claude 2发布", 1500, "Anthropic, 长文本"),
    ("2023-12-15", "Gemini Ultra发布", 1600, "Google, 首个多模态超越GPT-4"),
    ("2024-02-15", "Sora + Gemma发布", 1750, "OpenAI视频生成 + Google开源"),
    ("2024-03-15", "Claude 3 Opus", 1900, "Anthropic, 全面超越GPT-4"),
    ("2024-05-15", "GPT-4o发布", 2100, "OpenAI, 原生多模态+实时语音"),
    ("2024-07-15", "Claude 3.5 Sonnet", 2250, "Anthropic, 编程能力领先"),
    ("2024-09-15", "o1-preview发布", 2500, "OpenAI, 推理能力突破"),
    ("2024-12-15", "DeepSeek V3 + Gemini 2.0", 2750, "中国开源逼近闭源 + Google"),
    ("2025-01-15", "o3发布", 3000, "OpenAI, ARC AGI基准突破"),
    ("2025-02-15", "Grok 3发布", 3100, "xAI, 合成数据训练突破"),
    ("2025-03-15", "Claude Opus 4", 3250, "Anthropic, 长文本+深度推理"),
    ("2025-05-15", "Gemini 2.5 Pro", 3400, "Google, 推理能力新高度"),
    ("2025-07-15", "Llama 4 + Mistral Large 3", 3500, "Meta+欧洲, 开源爆发"),
    ("2025-09-15", "GPT-5发布", 3900, "OpenAI, 百万token上下文"),
    ("2025-11-15", "Claude Opus 4.5", 4100, "Anthropic, 科学推理领先"),
    ("2026-01-15", "DeepSeek V4预告 + Gemini 3.0", 4250, "中国模型突破+Google"),
    ("2026-02-15", "Grok 4 + 智谱GLM-5发布", 4400, "xAI + 中国并行突破"),
    ("2026-03-01", "DeepSeek V4上线", 4600, "SWE-Bench 83.7%, API价格1/20"),
    ("2026-03-15", "Claude Mythos/Capybara泄露", 4700, "Anthropic第四档"),
    ("2026-04-15", "GPT-5.5 + GLM-5.1 + Claude 4.7", 4800, "三巨头密集发布期"),
    ("2026-05-16", "Claude Opus 4.8 + Qwen3.7 Max", 4850, "Anthropic+阿里同期发布"),
    ("2026-06-01", "GPT-5.6 + Gemini 3.1 Pro", 4900, "OpenAI+Google最新旗舰"),
]

# ── 中美大模型能力差距锚点 ──────────────────────────────────────
# parity 综合  = 纯能力×0.5 + 性价比×0.25 + 开源生态×0.15 + 创新度×0.1
# 纯能力基于国际公认基准加权平均:
#   - LMSYS Chatbot Arena Elo (人类盲测偏好, 权重35%)
#   - MMLU (知识理解, 权重20%)
#   - GPQA (研究生级科学推理, 权重20%)
#   - SWE-bench (软件工程, 权重15%)
#   - MATH/AIME (数学推理, 权重10%)
# 数据来源: lmarena.ai, llm-stats.com, benchlm.ai (2026.06)
#
# 2026.06 国际基准综合纯能力比:
#   DeepSeek V4.1 Pro = US TOP × 96% (Arena Elo 1410/1465)
#   Qwen 3.7         = US TOP × 95% (Arena Elo 1400/1465)
#   GLM-6            = US TOP × 93%
#   Kimi K2.6        = US TOP × 92%

CN_US_PARITY = [
    # (日期, 事件, 综合parity, 纯能力, 性价比, 开源, 创新, 说明)
    ("2022-12-01", "GPT-4 vs 百度文心",        12, 12, 15,  8, 10, "中国大模型刚起步"),
    ("2023-03-15", "GPT-4正式 vs 阿里通义",     16, 15, 20, 12, 15, "中国快速跟进"),
    ("2023-07-15", "Claude 2 vs 百度ERNIE3.5", 20, 18, 25, 18, 20, "开源路径开始"),
    ("2023-12-15", "Gemini vs 百度ERNIE4.0",   25, 22, 30, 25, 25, "中国模型开始追赶"),
    ("2024-02-15", "Sora vs 字节豆包",         28, 25, 35, 28, 28, "多模态差距明显"),
    ("2024-05-15", "GPT-4o vs 智谱GLM-4",      32, 30, 38, 32, 30, "多模态差距缩小"),
    ("2024-07-15", "Claude3.5 vs 阿里Qwen2",   36, 34, 40, 38, 34, "中国开源路径确立"),
    ("2024-09-15", "o1 vs 智谱GLM-4/Qwen2.5",  40, 38, 45, 42, 38, "推理能力追赶"),
    ("2024-12-15", "DeepSeek V3发布",           60, 50, 85, 72, 60, "性价比碾压+开源爆发"),
    ("2025-02-15", "Grok 3 vs DeepSeek V3",    56, 48, 78, 68, 55, "纯能力差距缩小中"),
    ("2025-03-15", "Claude Opus4 vs DeepSeek", 58, 50, 78, 68, 58, "旗舰对决"),
    ("2025-05-15", "Gemini2.5 vs Qwen2.5",     60, 52, 78, 70, 56, "中国开源生态壮大"),
    ("2025-07-15", "Llama4 vs Qwen3/GLM-4.5",  62, 55, 78, 72, 60, "开源生态同步发展"),
    ("2025-09-15", "GPT-5 vs Kimi K2",          58, 50, 75, 68, 58, "GPT-5百万token拉开差距"),
    ("2025-11-15", "Claude4.5 vs GLM-5预览",   62, 55, 78, 70, 62, "长文本中国追赶"),
    ("2026-01-15", "Gemini3.0 vs DS V4预告",   66, 60, 82, 72, 65, "中国加速追赶"),
    ("2026-02-15", "Grok 4 vs 智谱GLM-5",      70, 64, 84, 75, 70, "中国多路径并进"),
    ("2026-03-01", "DeepSeek V4上线",           78, 72, 96, 88, 82, "Arena Elo 96%, 性价比碾压"),
    ("2026-03-15", "Capybara vs DeepSeek V4",   76, 70, 94, 86, 80, "美国反超纯能力"),
    ("2026-04-15", "GPT-5.5 vs GLM-5.1/DS-V4", 77, 72, 92, 84, 80, "中美多模型交替"),
    ("2026-06-01", "GPT-5.6 vs Qwen3.7/GLM-6", 78, 74, 90, 84, 80, "Elo 96%/95%/93%, 性价比88%"),
]


def load_history(path="data/curated/aidi_history.json"):
    """加载历史数据"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_benchmarks(path="data/benchmarks/benchmarks.json"):
    """加载外部基准测评数据"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"records": []}


def interpolate_aic(date_str):
    """在锚点间插值, 返回任意日期的AIC估值"""
    from datetime import datetime as dt
    target = dt.strptime(date_str, "%Y-%m-%d")
    
    # 找到目标日期前后的锚点
    before, after = None, None
    for d, _, aic, _ in AIC_ANCHORS:
        d_obj = dt.strptime(d, "%Y-%m-%d")
        if d_obj <= target:
            before = (d_obj, aic)
        if d_obj >= target and after is None:
            after = (d_obj, aic)
    
    if before is None:
        before = (dt.strptime(AIC_ANCHORS[0][0], "%Y-%m-%d"), AIC_ANCHORS[0][2])
    if after is None:
        after = (dt.strptime(AIC_ANCHORS[-1][0], "%Y-%m-%d"), AIC_ANCHORS[-1][2])
    
    if before[0] == after[0]:
        return before[1]
    
    # 线性插值
    total = (after[0] - before[0]).days
    elapsed = (target - before[0]).days
    ratio = elapsed / total if total > 0 else 0
    return round(before[1] + (after[1] - before[1]) * ratio)


def build_aic_timeseries(start="2022-12-01", end="2026-06-21"):
    """构建从基线的完整AIC时间序列 (每半月)"""
    from datetime import datetime as dt, timedelta
    
    start_d = dt.strptime(start, "%Y-%m-%d")
    end_d = dt.strptime(end, "%Y-%m-%d")
    
    periods = []
    current = start_d.replace(day=1)
    
    while current <= end_d:
        for day in [1, 16]:
            d = current.replace(day=min(day, 28))
            if d < start_d or d > end_d:
                continue
            if day == 1:
                next_d = current.replace(day=16)
            else:
                next_d = (current.replace(month=current.month+1) if current.month < 12 
                         else current.replace(year=current.year+1, month=1))
            period_end = next_d - timedelta(days=1)
            
            aic = interpolate_aic(d.strftime("%Y-%m-%d"))
            
            periods.append({
                "date": d.strftime("%Y-%m-%d"),
                "aic": aic
            })
        
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return periods


def build_full_timeseries():
    """构建完整AIC+AIDI时间序列"""
    periods = build_aic_timeseries()
    
    for i, p in enumerate(periods):
        if i == 0:
            p["aidi"] = 100  # 初始速度 100km/h (GPT-4发布)
        else:
            # AIDI = 每半月AIC增量
            p["aidi"] = round(p["aic"] - periods[i-1]["aic"], 1)
    
    return periods


def calc_aidi_from_aic(aic_current, aic_previous):
    """AIDI = AIC增量 (每半月)"""
    return round(aic_current - aic_previous, 1)


def bayesian_predict_aic(aics):
    """基于AIC历史预测下期AIC"""
    if len(aics) < 2:
        return {"predicted": None}
    
    deltas = [aics[i] - aics[i-1] for i in range(1, len(aics))]
    
    # 加权趋势 (最近权重更大)
    w = [0.1, 0.15, 0.2, 0.25, 0.3][:len(deltas)]
    w = [x/sum(w) for x in w]
    avg_delta = sum(d * w_i for d, w_i in zip(deltas, w))
    
    # 波动性
    if len(deltas) > 1:
        variance = sum((d - sum(deltas)/len(deltas))**2 for d in deltas) / len(deltas)
        volatility = math.sqrt(variance)
    else:
        volatility = abs(avg_delta) * 0.5 if avg_delta else 5
    
    latest = aics[-1]
    predicted = round(latest + avg_delta)
    ci = round(1.96 * max(volatility, 5))
    
    return {
        "predicted": predicted,
        "ci_low": max(1, predicted - ci),
        "ci_high": predicted + ci,
        "confidence": round(max(0, min(100, 100 - ci / max(predicted, 1) * 100))),
        "trend_delta": round(avg_delta, 1),
        "volatility": round(volatility, 1),
        "predicted_aidi": round(avg_delta, 1)
    }


def calc_validation_aic(periods):
    """校验函数: 回头看预测精度"""
    if len(periods) < 3:
        return {"F": None}
    
    aics = [p["aic"] for p in periods]
    errors = []
    
    for i in range(2, len(periods)):
        pred = bayesian_predict_aic(aics[:i])
        if pred["predicted"]:
            actual = aics[i]
            errors.append({
                "date": periods[i]["date"],
                "actual": actual,
                "predicted": pred["predicted"],
                "error": actual - pred["predicted"],
                "in_ci": pred["ci_low"] <= actual <= pred["ci_high"]
            })
    
    if not errors:
        return {"F": None}
    
    mape = sum(abs(e["error"]) for e in errors) / sum(e["actual"] for e in errors)
    in_ci_rate = sum(1 for e in errors if e["in_ci"]) / len(errors)
    calibration = 1 - abs(in_ci_rate - 0.95)
    F = round((1 - mape) * 0.6 + calibration * 0.4, 4)
    
    return {
        "F": F,
        "mape": round(mape, 4),
        "in_ci_rate": round(in_ci_rate, 2),
        "calibration": round(calibration, 4),
        "total_checks": len(errors),
        "errors": errors[-5:]
    }


def calc_cn_us_parity():
    """计算中美AI能力差距时间序列"""
    from datetime import datetime as dt, timedelta
    
    # 构建每半月的parity插值
    start = dt.strptime("2022-12-01", "%Y-%m-%d")
    end = dt.strptime("2026-06-16", "%Y-%m-%d")
    
    parities = []
    current = start.replace(day=1)
    
    while current <= end:
        for day in [1, 16]:
            d = current.replace(day=min(day, 28))
            if d < start or d > end:
                continue
            
            ds = d.strftime("%Y-%m-%d")
            
            # 找前后锚点插值 (新格式: 7字段, 第3=综合parity)
            before, after = None, None
            for row in CN_US_PARITY:
                date_str = row[0]
                p = row[2]  # 综合parity
                dd = dt.strptime(date_str, "%Y-%m-%d")
                if dd <= d:
                    before = (dd, p)
                if dd >= d and after is None:
                    after = (dd, p)
            
            if before is None:
                before = after or (d, 15)
            if after is None:
                after = before
            
            if before[0] == after[0]:
                parity = before[1]
            else:
                total_days = (after[0] - before[0]).days
                elapsed = (d - before[0]).days
                ratio = elapsed / total_days if total_days > 0 else 0
                parity = round(before[1] + (after[1] - before[1]) * ratio)
            
            parities.append({"date": ds, "parity": parity})
        
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return parities


def full_run():
    """全流程运行"""
    # 1. 构建完整时间序列
    periods = build_full_timeseries()
    parities = calc_cn_us_parity()
    
    current = periods[-1]
    baseline = periods[0]
    current_parity = parities[-1]
    
    print(f"[AIC基线] {baseline['date']}: AIC={baseline['aic']}")
    print(f"[AIC当前] {current['date']}: AIC={current['aic']} (较基线 +{current['aic'] - baseline['aic']})")
    print(f"[AIDI当前] {current['date']}: AIDI={current['aidi']} (最近半月增量)")
    print(f"[中美差距] {current_parity['date']}: 中国能力=美国{current_parity['parity']}%")
    
    # 2. 中美差距曲线 (含四维分解)
    print(f"\n[中美AI能力差距演变]")
    print(f"{'日期':<15} {'综合':>5} {'纯能力':>6} {'性价比':>6} {'开源':>5} {'创新':>5}")
    print("-" * 48)
    for p in parities:
        if p['date'] in [a[0] for a in CN_US_PARITY]:
            row = next((a for a in CN_US_PARITY if a[0] == p['date']), None)
            if row:
                print(f"{p['date']:<15} {row[2]:>3}%  {row[3]:>3}%  {row[4]:>3}%  {row[5]:>3}%  {row[6]:>3}%  ← {row[1][:18]}")
    
    # 3. 曲线摘要
    print(f"\n[AIC能力曲线]")
    print(f"{'日期':<15} {'AIC':>6} {'AIDI':>8} {'事件':<20}")
    print("-" * 55)
    for p in periods:
        if p['date'] in [a[0] for a in AIC_ANCHORS]:
            event = next((a[1] for a in AIC_ANCHORS if a[0] == p['date']), "")
            marker = "★" 
        else:
            event = ""
            marker = " "
        print(f"{p['date']:<15} {p['aic']:>6} {p['aidi']:>+8} {marker}{event[:18]}")
    
    # 4. 下期预测
    aics = [p["aic"] for p in periods]
    pred = bayesian_predict_aic(aics)
    if pred["predicted"]:
        print(f"\n[下期AIC预测] {pred['predicted']} (CI: {pred['ci_low']}~{pred['ci_high']}, 置信度{pred['confidence']}%)")
        print(f"[下期AIDI预测] {pred['predicted_aidi']:+.1f}")
    
    # 5. 校验
    val = calc_validation_aic(periods)
    if val["F"]:
        print(f"\n[校验] F={val['F']}, MAPE={val['mape']*100:.1f}%, 区间命中率={val['in_ci_rate']*100:.0f}%")
    
    # 6. 报告
    report = {
        "baseline": {"date": baseline["date"], "aic": baseline["aic"]},
        "current": {"date": current["date"], "aic": current["aic"], "aidi": current["aidi"]},
        "aic_gain": current["aic"] - baseline["aic"],
        "cn_us_parity": current_parity,
        "total_periods": len(periods),
        "prediction": pred,
        "validation": val,
        "anchors": [{"date": a[0], "event": a[1], "aic": a[2]} for a in AIC_ANCHORS]
    }
    
    # 中美差距修正系数
    p = current_parity["parity"]
    # 如果中国能力只有美国的78%, 则全球AIC实际应略低于美国AIC
    # 修正: 全球AIC ≈ 美国AIC × (0.5 + 中国占比/2)
    # 当parity=100时, 全球AIC=美国AIC; parity=0时, 全球AIC=美国AIC×0.5
    correction = 0.5 + p / 200
    print(f"\n[中美修正系数] {correction:.3f} (parity={p}% → 全球AIC=美国AIC×{correction:.3f})")
    print(f"[修正后全球AIC] {round(current['aic'] * correction)}")
    
    report_path = f"reports/aidi_report_{datetime.now().strftime('%Y%m%d')}.json"
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n[报告] 已保存: {report_path}")
    
    return report


if __name__ == "__main__":
    full_run()
