# AI Development Index (AIDI)

AI发展指数 — 从2022年12月GPT发布至今的定量追踪。

**基线**: AIC=1000, AIDI=100km/h (2022-12-01, GPT发布日)  
**更新频率**: 每半月  
**评分制**: AIDI 0-200km/h (发展速度), AIC ≥ 0 (能力总量)

## 设计原理

- **AIDI** (AI Development Index) = 发展速度 (类比km/h)
  - 六维加权评分: 模型/硬件/算法/商业/应用/开源
  - 基线100km/h (GPT-4发布水平)
  - 每期波动反映发展快慢变化

- **AIC** (AI Capability Index) = 能力总量 (绝对值)
  - AIDI的积分累计
  - 只升不降, 反映AI绝对能力增长
  - 与外部基准测评校准

- **校验函数 F** = 预测精度跟踪
  - MAPE偏差率 + 置信区间命中率 + 校准度
  - 逐期积累经验, 修正预测模型

## 仓库结构

```
├── data/raw/          ← 原始搜索数据 (不可修改)
├── data/curated/      ← 加工后AIDI/AIC时间序列
├── data/benchmarks/   ← 外部基准测评数据
├── engine/            ← AIDI/AIC计算引擎
├── skills/            ← 搜索技能定义
├── reports/           ← 每期分析报告
└── .github/workflows/ ← Actions采集工作流
```

## 使用

```bash
# 本地运行引擎分析最新数据
python engine/aidicore.py

# GitHub Actions: 采集一个周期
→ Actions → AIDI 数据采集 → 输入日期范围

# GitHub Actions: 批量回填86期
→ Actions → AIDI 批量回填
```
