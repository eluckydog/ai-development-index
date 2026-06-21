# AI Development Index (AIDI)

**AI能力的"GDP" —— 一个可检验、可追溯、可校准的全球AI发展定量追踪系统。**

这不是大模型公司的PR稿汇总。是**从物理隐喻出发，用数学工具建模，用真实数据注入，用后见之明校准**的工程化指数。数据+算法=唯一确定输出，换任何模型跑都得到相同结果。

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIDI 引擎管线 (v3/v4/v5/v6)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  v3 能力指数                               v4 发展速度            │
│  ┌─────────────────────────┐       ┌──────────────────────┐    │
│  │ 六维能力向量             │       │ 四维速度 (DVI)        │    │
│  │  + 跨维度交互乘数        │       │  + PSO权重优化        │    │
│  │  → AIC (能力总量)       │       │  + HMM状态检测        │    │
│  └──────────┬──────────────┘       │  + 卡尔曼平滑         │    │
│             │                      └──────────────────────┘    │
│             ▼                                                   │
│  v5 预测引擎 (算法层 × 语境层)          v6 能力演化引擎           │
│  ┌─────────────────────────┐       ┌──────────────────────┐    │
│  │ Holt-Winters + ARIMA    │       │ 三阶段演化框架        │    │
│  │  + 指数拟合 + Bootstrap │       │  + Granger因果检验    │    │
│  │  + 语境调校因子         │       │  + S曲线相变预测      │    │
│  │  → 6条断言预测          │       │  → 4条演化断言        │    │
│  └─────────────────────────┘       └──────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 引擎

| 模块 | 功能 | 方法 |
|:----|:----|:----|
| `aidicore_v3.py` | **能力指数 (AIC)** | 六维向量(智力/感官/行动力/编程/知识/生态) + 跨维度交互乘数 |
| `aidicore_v4.py` | **发展速度 (DVI)** | 四维(硬件/效率/定价/渗透) + PSO + HMM + 卡尔曼平滑 |
| `aidicore_v5.py` | **预测引擎** | 双层架构: 算法层(统计) × 语境层(LLM调校) |
| `aidicore_v6.py` | **能力演化** | 三阶段框架 + Granger因果 + S曲线 + Phase Transition |
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

| 日期 | AIC | AIDI | 意义 |
|:---:|:---:|:----:|:----|
| 2022-12-01 | **1,000** | **100** | ChatGPT上线, **基线** |
| 2024-01-01 | **10,333** | +307 | 一年后, 能力×10 |
| 2025-01-01 | **29,974** | +1,128 | o3+DeepSeek R1, 能力×30 |
| 2026-01-01 | **83,769** | +4,513 | GPT-5时代, 能力×84 |
| 2026-06-16 | **222,923** | +11,744 | **当前, 能力×223** |

中美综合parity: **78%** (纯能力74%, 性价比90%, 开源84%, 创新80%)

---

## 快速使用

```bash
pip install -r requirements.txt

# 能力指数 (AIC)
python engine/aidicore_v3.py

# 发展速度 (DVI)
python engine/aidicore_v4.py

# 预测引擎 (算法层 × 语境层)
python engine/aidicore_v5.py                     # 双层融合
python -c "from engine.aidicore_v5 import AIDIPredictor; AIDIPredictor().run(algorithm_only=True)"  # 纯算法

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

### AIC (AI Capability)

```
AIC = 六维能力向量平均值 + 跨维度交互效应

基础项:       base = Σ(维度分) / 6
交互项:       synergy = Σ(s_ij × min(vi,vj)/1000 × vi×vj/1e6 × 500)
```

交互项用**弱维度约束×乘积效应**来建模协同——链子的强度取决于最弱的一环。

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

### AIC聚合预测

| 日期 | 聚合AIC | 95% CI |
|:----|:------:|:------:|
| 2026-12-16 | **386,773** | 359,518 ~ 414,055 |
| 2027-07-01 | **665,914** | 466,140 ~ 865,688 |

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
│   ├── aidicore_v3.py         # 能力指数
│   ├── aidicore_v4.py         # 发展速度
│   ├── aidicore_v5.py         # 预测引擎 (双层架构)
│   ├── aidicore_v6.py         # 能力演化 (三阶段+Granger)
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
