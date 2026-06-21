#!/usr/bin/env python3
"""
AIDI 首次完整基线确认 + 当前点位确认
基线: 2022-12-01 (GPT发布), AIC=1000, AIDI=100km/h
当前: 2026-06-15 最新数据
"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

repo = r'C:\Users\13918\WorkBuddy\2026-06-18-21-18-46\ai-development-index'
sys.path.insert(0, os.path.join(repo, 'engine'))
from aidicore import *

print('=' * 65)
print('AIDI 首次基线确认 & 当前点位确认')
print('=' * 65)

# ── 加载现有历史数据 ──
history_path = os.path.join(repo, 'data/curated/aidi_history.json')
benchmarks_path = os.path.join(repo, 'data/benchmarks/benchmarks.json')

data = load_history(history_path)
history = data['history']

# ── 基线设定 ──
# 2022-12-01: GPT发布
# AIDI = 100 (初始速度)
# AIC = 1000 (初始能力)
print(f'\n[基线] 2022-12-01: GPT-4发布')
print(f'  AIDI = 100 (初始发展速度)')
print(f'  AIC  = 1000 (初始能力基线)')

# ── 添加基线期 ──
baseline = {
    'date': '2022-12-01',
    'cycle': 0,
    'dimensions': {'model': 100, 'hardware': 100, 'algorithm': 100, 'business': 100, 'adoption': 100, 'opensource': 100},
    'aidi': 100,
    'aic': 1000,
    'direction': 'stable',
    'summary': 'GPT-4发布, AI发展指数基线'
}
history.insert(0, baseline)

# ── 计算AIC ──
aics = calc_aic(history)
for i, h in enumerate(history):
    h['aic'] = aics[i]

# ── 当前点位 ──
current = history[-1]
print(f'\n[当前] {current["date"]}:')
print(f'  AIDI = {current["aidi"]}')
print(f'  AIC  = {current["aic"]}')
print(f'  方向 = {current["direction"]}')
print(f'  摘要 = {current["summary"]}')

# ── 完整曲线 ──
print(f'\n[完整AIC曲线]')
print(f'{"日期":<15} {"AIDI":>6} {"AIC":>8} {"方向":<12}')
print('-' * 45)
for h in history:
    print(f'{h["date"]:<15} {h["aidi"]:>6} {h["aic"]:>8.1f} {h["direction"]:<12}')

# ── 基准对齐 ──
print(f'\n[基准对齐]')
benchmarks = load_benchmarks(benchmarks_path)
calib = calibrate_aic(history, benchmarks)
if calib['calibrations']:
    for c in calib['calibrations']:
        print(f'  {c["date"]}: AIC模型={c["aic_model"]:.0f} AIC实测={c["aic_actual"]} gap={c["gap"]:+.0f} ({c["gap_pct"]*100:+.1f}%)')
    print(f'  调整因子: {calib.get("adjustment_factor", "N/A")}')

# ── 预预测下期 ──
print(f'\n[下期预测]')
# 去掉基线期做预测
pred_history = history[1:]  # 只用实际数据期
pred = bayesian_predict(pred_history)
if pred['predicted']:
    print(f'  下期AIDI预计: {pred["predicted"]} (CI: {pred["ci_low"]}~{pred["ci_high"]})')
    print(f'  置信度: {pred["confidence"]}%')
    print(f'  趋势变化: {pred["trend_delta"]:+.1f}')

# ── 校验 ──
print(f'\n[预测校验]')
val = calc_validation(pred_history)
if val['F']:
    print(f'  F值: {val["F"]}')
    print(f'  MAPE: {val["mape"]*100:.2f}%')
    print(f'  区间命中率: {val["in_ci_rate"]*100:.0f}%')

# ── 保存报告 ──
report = {
    'baseline': {'date': '2022-12-01', 'aidis': 100, 'aic': 1000},
    'current': {'date': current['date'], 'aidi': current['aidi'], 'aic': current['aic']},
    'aic_gain': round(current['aic'] - 1000, 1),
    'total_periods': len(history) - 1,
    'prediction': pred,
    'calibration': calib,
    'validation': val,
}

report_path = os.path.join(repo, 'reports/aidi_first_run.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f'\n[报告] 已保存: {report_path}')
print('\n[完成]')
