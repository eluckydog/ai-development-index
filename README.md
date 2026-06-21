# AI Development Index (AIDI)

**AI能力的"GDP" —— 一个可检验、可追溯、可校准的全球AI发展定量追踪系统。**

这不是大模型公司的PR稿汇总。是**从物理隐喻出发，用数学工具建模，用真实数据注入，用后见之明校准**的工程化指数。数据+算法=唯一确定输出，换任何模型跑都得到相同结果。

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│              AIDI 引擎管线 (v3/v4/v5/v6/v7)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  v3 (旧) 能力指数                       v7 (新) 一阶齐次AIC          │
│  ┌──────────────────────┐       ┌────────────────────────────┐     │
│  │ 六维向量+交互乘数     │       │ Cobb-Douglas几何平均       │     │
│  │ → AIC, 但O(n³)膨胀   │       │ + 可选R归一化交互          │     │
│  │ 基线AIC=1000          │       │ → AIC, 一阶齐次            │     │
│  │ 当前: 222,923 (×223)  │       │ 基线AIC=1000               │     │
│  └──────────────────────┘       │ 当前: 78,217 (×78)         │     │
│                                 └────────────────────────────┘     │
│  v4 发展速度                            v5 预测引擎 (三层)         │
│  ┌──────────────────────┐       ┌────────────────────────────┐     │
│  │ DVI(硬件/效率/定价/渗透)│       │ 算法层→相变感知层→语境层   │     │
│  │ + PSO+HMM+卡尔曼      │       │ → 三情景+6条断言           │     │
│  └──────────────────────┘       └────────────────────────────┘     │
│                                                                     │
│  v6 能力演化引擎                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 三阶段框架(Phase 1封闭→Phase 2感官→Phase 3世界模型)           │   │
│  │ + Granger因果检验 + S曲线相变预测 → 4条演化断言               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 引擎

| 模块 | 功能 | 方法 |
|:----|:----|:----|
| `aidicore_v3.py` | **能力指数 (旧)** | 六维向量 + 交互乘数 (O(n³)膨胀, 待迁移) |
| `aidicore_v4.py` | **发展速度 (DVI)** | 四维(硬件/效率/定价/渗透) + PSO + HMM + 卡尔曼平滑 |
| `aidicore_v5.py` | **预测引擎** | 三层架构: 算法层(统计) × 相变感知层(regime) × 语境层(LLM调校) |
| `aidicore_v6.py` | **能力演化** | 三阶段框架 + Granger因果 + S曲线 + Phase Transition |
| `aidicore_v7.py` | **能力指数 (新, 推荐)** | Cobb-Douglas几何平均 + 0次齐次R交互 (一阶齐次) |
| `backtest.py` | **历史回测** | 5个截断点 × 4个预测期 × 因果稳定性检验 |

### 数据

| 路径 | 内容 | 不可变? |
|:----|:----|:-------:|
| `data/raw/YYYY-MM-DD/` | 86期原始搜索数据 | ✅ 不可变 |
| `data/curated/dim_scores.json` | 六维能力评分 (可校准) | 可追加 |
| `data/curated/calibration_log.json` | 每次校准记录 | 可追加 |
| `data/benchmarks/benchmarks.json` | 外部基准 (Arena Elo/MMLU/GPQA) | 仅增 |

---

## 基线数据

| 日期 | AIC (v3旧) | AIC (v7新,推荐) | AIDI | 意义 |
|:---:|:----------:|:--------------:|:----:|:----|
| 2022-12-01 | **1,000** | **1,000** | **100** | ChatGPT上线, **基线** |
| 2024-01-01 | **10,333** | **12,644** | +307 | 一年后 |
| 2025-01-01 | **29,974** | **31,083** | +1,128 | o3+DeepSeek R1 |
| 2026-01-01 | **83,769** | **52,708** | +4,513 | GPT-5时代 |
| 2026-06-16 | **222,923** | **78,217** | +11,744 | **当前** |

> v7公式: `AIC = A × (Π dim_i)^(1/6)` (Cobb-Douglas几何平均, 一阶齐次)
> 
> v3旧公式存在O(n³)交互膨胀, v7修正后增长×78(而非×223), 数学自洽, 与HDI接轨。

中美综合parity: **78%** (纯能力74%, 性价比90%, 开源84%, 创新80%)

---

## 快速使用

```bash
pip install -r requirements.txt

# v7 能力指数 (新, 推荐 — Cobb-Douglas几何平均, 一阶齐次)
python engine/aidicore_v7.py

# v3 能力指数 (旧)
python engine/aidicore_v3.py

# 发展速度 (DVI)
python engine/aidicore_v4.py

# 预测引擎 (三层: 算法 x 相变感知 x 语境)
python engine/aidicore_v5.py                     # 默认: 三层融合
python -c "from engine.aidicore_v5 import AIDIPredictor; AIDIPredictor().run(algorithm_only=True)"  # 纯算法(可复现)

# 能力演化 (Granger因果 + Phase Transition)
python engine/aidicore_v6.py

# 历史回测
python engine/backtest.py
```

### 作为模块导入

```python
from engine.aidicore_v5 import AIDIPredictor
from engine.aidicore_v6 import CapabilityEvolutionEngine, granger_causality

# 预测AIC
pred = AIDIPredictor()
aic_12m = pred.predict_aic(horizon=12)  # → {aggregate, ci_95, ...}

# 断言预测
assertions = pred.get_assertions(algorithm_only=False)
for a in assertions:
    print(f"{a['statement']}: 算法{a['algorithm_layer']['base_probability']*100:.0f}% → 融合{a['fused_probability']*100:.0f}%")

# Granger因果检验
evo = CapabilityEvolutionEngine()
result = granger_causality(evo.scores["text_llm"], evo.scores["image"], max_lag=4)
print(f"text_llm → image: 因果{'成立' if result['causal'] else '不成立'} (p={result['best_p_value']:.4f})")
```

---

## 设计原理

### AIC (AI Capability, v7新公式 — 推荐)

```
AIC = A × (Π dim_i)^(1/6)   (Cobb-Douglas几何平均, 等权α_i=1/6)

可选的归一化交互: AIC × (1 + η × Σ s_ij × R_ij)
R(v_i, v_j) = min(v_i, v_j)² / (v_i² + v_j²)   ← 0次齐次, 不随量级爆炸
```

**设计依据**: 一阶齐次性——所有维度翻k倍时AIC也翻k倍。这是经济学(HDI、Cobb-Douglas生产函数)的标准约束。v3旧公式的交互项是O(n³), 导致×223中只有约×78来自真实能力提升。

### 校准机制

每次校准允许回溯修正历史评分，并记录修正原因到 `calibration_log.json`。这是**贝叶斯更新**在时间序列上的应用——新证据不仅更新当前期，也更新对历史的理解。

### 外部验证

| 验证项 | AIDI | 外部数据 | 偏差 |
|:------|:----|:--------|:---:|
| 硬件计算增长 | 30倍 | Epoch AI: 30倍 | **0%** |
| SWE-bench趋势 | 编程×2.5 | Stanford HAI: 60%→近100% | 趋势一致 |
| AI Agent趋势 | 行动力5→1580 | Stanford HAI: OSWorld 12%→66% | 趋势一致 |

完整对比: `reports/AIDI_vs_外部指数_2026年完整对比.md`

---

## 预测 (2026-06-21 生成)

### v5 断言预测 (算法×语境)

| # | 断言 | 算法基概率 | 融合概率 |
|:-|:----|:---------:|:--------:|
| 1 | Agent规模化爆发 (DAU>1亿) | 95% | 95% |
| 2 | 推理成本再降50% (<$0.08/1M) | 80% | 72% |
| 3 | 编程自改进成熟 (交互效应>8%) | 85% | 86% |
| 4 | CN-US综合parity→85%+ | 79% | 84% |
| 5 | AI普及率跨越65% | 50% | 44% |
| 6 | ≥3国出台AI能源法规 | 54% | 59% |

### v6 演化断言 (三阶段框架)

| ID | 断言 | 预测时间 | 概率 |
|:--|:----|:--------|:---:|
| E1 | 世界模型商用化(700分) | 2027年 | 31% |
| E2 | 通用物理操作(700分) | >2029 | 10% |
| E3 | Phase 2→3 相变 | 进行中 | 60% |
| E4 | 因果推理超越模式匹配 | 2028年 | 35% |

### AIC聚合预测 (v7新AIC)

| 日期 | 聚合AIC (v7) | 95% CI (v7) |
|:----|:-----------:|:-----------:|
| 2026-12-16 | **~135,000** | — |
| 2027-07-01 | **~233,000** | — |

> 注: v5预测引擎的AIC基数从v3切换为v7后, 预测值相应调整。

---

## 回测结果

| 截断点 | 平均偏差 | 最佳表现 |
|:------|:-------:|:--------|
| 2024-06-01 | **-1.4%** | +3月:-1.2%, +6月:-2.0%, +9月:-1.4%, +12月:-0.9% |
| 2025-01-01 | -7.4% | +3月:+2.5%, +6月:+2.4% |
| 2025-06-01 | -33.5% | GPT-5爆发导致结构性低估 |

完整回测: `engine/backtest.py` → `reports/aidi_backtest_report.json`

---

## 仓库结构

```
ai-development-index/
├── engine/                     # 引擎模块
│   ├── aidicore_v3.py         # 能力指数 (旧: 六维+交互乘数)
│   ├── aidicore_v4.py         # 发展速度
│   ├── aidicore_v5.py         # 预测引擎 (三层架构)
│   ├── aidicore_v6.py         # 能力演化 (三阶段+Granger)
│   ├── aidicore_v7.py         # 能力指数 (新推荐: CD几何平均)
│   ├── backtest.py            # 历史回测
│   └── period_search.py       # 数据获取
├── data/
│   ├── raw/                   # 原始数据 (不可变)
│   ├── curated/               # 策展数据 (可校准)
│   └── benchmarks/            # 外部基准
├── reports/                   # 报告输出
├── skills/                    # WorkBuddy skills
├── .github/workflows/         # CI/CD
├── .gitignore
├── requirements.txt
├── AGENT_MANUAL.md            # Agent操作手册
└── README.md
```

---

## 许可证

开源项目 · github.com/eluckydog/ai-development-index · 基线: 2022-12-01 ChatGPT上线
