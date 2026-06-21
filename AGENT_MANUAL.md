# AIDI Agent 操作手册

> 版本: v5+v6 双层引擎管线  
> 最后更新: 2026-06-21  
> 适用: 任何AI Agent或人类操作者

---

## 第一章：项目定位

### AIDI是什么？

AI Development Index (AIDI) 是一个**定量追踪全球AI技术发展**的工程化指数系统。类似GDP之于经济，AIDI用数学工具量化AI的能力总量(AIC)和发展速度(AIDI)。

**核心哲学**：
- 数据+算法 = 唯一确定输出
- 不依赖任何特定LLM模型的推理
- 可复现、可检验、可校准

### 谁会用这个系统？

| 角色 | 典型任务 | 主要模块 |
|:----|:--------|:--------|
| **数据分析师** | 查看当前AIC/AIDI、对比历史趋势 | v3, v4 |
| **预测研究员** | 生成未来预测、检验断言 | v5 |
| **战略分析师** | 理解AI能力演化路径、因果链 | v6 |
| **运维Agent** | 数据采集、校准、推送到GitHub | period_search, calibration |
| **审计Agent** | 三阶段检查、回测验证 | backtest |

---

## 第二章：引擎管线详解

### 2.1 v3 — 能力指数 (AIC)

**文件**: `engine/aidicore_v3.py`

**功能**: 计算AI能力总量，6个维度 + 跨维度交互乘数。

**六个能力维度**：

| 维度 | 英文ID | 含义 | 基线(2022-12) | 当前(2026-06) |
|:----|:------|:----|:------------:|:------------:|
| 语言智力 | intelligence | LLM推理/知识 | 100 | 2,750 |
| 多模态感知 | multimodal | 看/听/说/生成 | 10 | 1,470 |
| 智能体行动力 | agent | 工具/规划/自主 | 5 | 1,580 |
| 编程自改进 | programming | 代码生成/自我改进 | 80 | 2,150 |
| 知识系统 | knowledge | RAG/向量库/记忆 | 10 | 1,380 |
| 生态基础设施 | ecosystem | API/成本/框架 | 30 | 1,450 |

**使用方法**：
```python
from engine.aidicore_v3 import build_timeseries_v3, run_full_v3

# 获取全量时间序列
periods = build_timeseries_v3(normalize=True)
# periods[0] = 基线, periods[-1] = 当前

# 运行完整报告
report = run_full_v3()
```

**数据来源**: `data/curated/dim_scores.json` 中的86期六维评分。

### 2.2 v4 — 发展速度 (DVI)

**文件**: `engine/aidicore_v4.py`

**功能**: 追踪AI发展速度，使用PSO + HMM + 卡尔曼平滑。

**四个速度维度**：
- 硬件: GPU/带宽/HBM进展
- 效率: 量化/蒸馏/算法优化
- 定价: API成本 (log2尺度)
- 渗透: 采用率/普及度

**状态检测 (HMM)**：
- 爆发期 (Explosive): DVI增速 > 100
- 加速期 (Accelerating): 40-100
- 平台期 (Plateau): 10-40
- 减速期 (Decelerating): < 10

**使用方法**：
```bash
python engine/aidicore_v4.py
```

### 2.3 v5 — 预测引擎 (双层架构)

**文件**: `engine/aidicore_v5.py`

**这是最常用的高级引擎。** 双层架构:

```
算法层 (统计模型)             语境层 (LLM调校因子)
─────────────────         ─────────────────
Holt-Winters 指数平滑      风险/加速因素识别
ARIMA(p,d,q)              历史类比校准
指数增长拟合                跨界知识注入
Logistic S-curve          政策/经济上下文
Bootstrap置信区间
         ↓                          ↓
         └────────── 融合 ──────────┘
               fused = base × Π(1 + dir × mag)
```

**核心API**：

```python
from engine.aidicore_v5 import AIDIPredictor

pred = AIDIPredictor()

# ── 模式A: 纯算法 (可复现, 推荐) ──
result = pred.run(algorithm_only=True)

# ── 模式B: 双层融合 (算法×语境) ──
result = pred.run(algorithm_only=False)

# ── 单次AIC预测 ──
aic = pred.predict_aic(horizon=12)
# 返回: {holtwinters, arima, exponential, aggregate, ci_95}

# ── 单维度预测 ──
agent = pred.predict_dimension("agent", horizon=25)
# 返回: {dim, current, predictions: [1682, 1728, ...]}

# ── 断言预测 ──
assertions = pred.get_assertions(algorithm_only=False)
for a in assertions:
    print(f"断言{a['id']}: {a['statement']}")
    print(f"  算法概率: {a['algorithm_layer']['base_probability']*100:.0f}%")
    print(f"  融合概率: {a['fused_probability']*100:.0f}%")
    for adj in a['context_layer']['adjustments']:
        print(f"  调校: {adj['factor']} ({'+' if adj['direction']>0 else '-'}{adj['magnitude']:.0%})")
    print(f"  可校验: {', '.join(a['verifiable_by'])}")
```

**6条断言输出格式**：

每条断言包含：
- `algorithm_layer.base_probability` — 纯统计概率
- `context_layer.adjustments[]` — 每项调校因子及其理由
- `fused_probability` — 融合后概率
- `verifiable_by[]` — 可校验的3个条件

### 2.4 v6 — 能力演化引擎

**文件**: `engine/aidicore_v6.py`

**功能**: 基于三阶段演化框架分析AI能力发展路径。

**三阶段框架**：

```
Phase 1: 封闭系统突破                    Phase 2: 感官解锁与融合
┌──────────────────────┐               ┌──────────────────────────┐
│ 算法/围棋 → 文字(LLM) │    Granger→   │ 语音 → 图片 → 视频       │
│ 规则明确, 边界清晰    │    p<0.001    │ 单一感官→统一多模态       │
│ 当前成熟度: 765/1000  │               │ 当前成熟度: 790/1000     │
└──────────────────────┘               └───────────┬──────────────┘
                                                    │ Granger p<0.05
                                                    ▼
                              Phase 3: 感知到模拟
                              ┌──────────────────────────────┐
                              │ 世界模型 → 空间推理 → 物理交互 │
                              │ 理解因果 → 预测未来 → 操作世界 │
                              │ 当前成熟度: 463/1000          │
                              └──────────────────────────────┘
```

**核心API**：

```python
from engine.aidicore_v6 import CapabilityEvolutionEngine, granger_causality

evo = CapabilityEvolutionEngine()

# ── 全量分析 ──
result = evo.run(horizon=25, save=True)

# ── Granger因果检验 ──
# 检验 text_llm 是否 Granger-cause image
result = granger_causality(evo.scores["text_llm"], evo.scores["image"], max_lag=4)
if result["causal"]:
    print(f"因果成立 (p={result['best_p_value']:.4f}, lag={result['best_lag']})")

# ── Phase Transition预测 ──
transitions = evo.predict_phase_transition(horizon=25)
for cap_id, info in transitions.items():
    print(f"{info['name']}: 当前{info['current_score']} → 700分预计{info['projected_700_date']}")

# ── 演化断言 ──
assertions = evo.generate_evolution_assertions(horizon=25)
```

**Granger结果解读**：

| p值 | 结论 | 行动 |
|:---|:----|:----|
| < 0.01 | 强因果 | 演化路径确认 |
| < 0.05 | 显著因果 | 路径可能成立 |
| < 0.10 | 弱显著 | 需要更多数据 |
| ≥ 0.10 | 不显著 | 路径不成立或数据不足 |

**跨相关性解读**：
- `peak_lag > 0`: x领先y（x的变化先于y发生）
- `peak_lag < 0`: y领先x
- `peak_lag = 0`: 同步运动

---

## 第三章：数据操作

### 3.1 数据文件结构

```
data/
├── raw/                    # 原始数据 (不可变!)
│   ├── 2022-12-01/
│   │   ├── _search.json   # WebSearch原始结果
│   │   └── _period.json   # 该期原始事件
│   ├── ...
│   └── 2026-06-16/
├── curated/                # 策展数据 (可校准)
│   ├── dim_scores.json    # 六维评分 (引擎读这个)
│   ├── calibration_log.json # 每次修正记录
│   └── complete_snapshot.json
└── benchmarks/
    └── benchmarks.json     # 外部基准
```

### 3.2 dim_scores.json 格式

```json
{
  "version": "v3",
  "dimensions": ["intelligence", "multimodal", "agent", "programming", "knowledge", "ecosystem"],
  "periods": {
    "2022-12-01": {
      "scores": {"intelligence": 100, "multimodal": 10, ...},
      "note": "ChatGPT上线, 编程有Copilot/Codex, 其他刚起步"
    },
    ...
  }
}
```

**评分准则**:
- 0-200: 概念/原型阶段
- 200-400: 萌芽期 (学术→早期商用)
- 400-600: 成长期 (可用但有限)
- 600-800: 成熟期 (稳定商用)
- 800-1000: 顶尖期 (接近理论上限)

### 3.3 校准流程

当新模型发布后:

1. 更新 `dim_scores.json` — 添加新期或修正历史期
2. 在 `calibration_log.json` 记录修正原因
3. 重新运行v3+v4验证
4. 重新运行v5+v6更新预测

---

## 第四章：回测 (Backtest)

### 4.1 为什么回测

回测是用历史数据检验预测引擎的唯一方法。AIDI回测在5个历史截断点运行v5引擎。

### 4.2 运行回测

```bash
python engine/backtest.py
```

输出在 `reports/aidi_backtest_report.json`。

### 4.3 解读回测结果

**v5 AIC预测**：
- 计算各算法(Holt-W, ARIMA, 指数, 聚合)在每个截断点的偏差百分比
- 检查实际值是否在95%置信区间内
- 按预测期和截断点分组统计

**v6 因果稳定性**：
- 检查Granger因果结论是否随数据量增加而稳定
- 记录每条因果链"最早成立"的时间点

---

## 第五章：常见任务速查

### 任务1: 查看当前AIC和AIDI

```bash
python engine/aidicore_v3.py
```

输出中找"当前"行。

### 任务2: 生成2026年底预测

```python
from engine.aidicore_v5 import AIDIPredictor
pred = AIDIPredictor()
result = pred.run(algorithm_only=False, save=True)
# reports/aidi_v5_predictions.json
```

### 任务3: 检验"文字→图片"因果链

```python
from engine.aidicore_v6 import CapabilityEvolutionEngine, granger_causality
evo = CapabilityEvolutionEngine()
r = granger_causality(evo.scores["text_llm"], evo.scores["image"], max_lag=4)
print(f"因果成立: {r['causal']}, p={r['best_p_value']:.4f}")
```

### 任务4: 检查预测是否准确 (回测)

```bash
python engine/backtest.py 2>&1 | grep "最准"
```

### 任务5: 比较AIDI与外部指数

查看 `reports/AIDI_vs_外部指数_2026年完整对比.md`

---

## 第六章：注意事项

### 已知局限

1. **早期数据不足**: 2023年只有12-24期数据时，ARIMA和指数拟合会发散。此时只相信Holt-Winters。
2. **结构性断点**: GPT-5/Capybara爆发(2025-08)导致所有模型系统性低估~59%。这是scaling law跳变，非模型能预测。
3. **世界模型数据稀疏**: v6中world_model/physical_interaction只有2年数据，S曲线拟合不稳定。
4. **编码问题**: 脚本输出含Unicode箭头/符号时，在GBK终端会报错。所有关键数据在JSON中，不受终端编码影响。

### 最佳实践

1. **先跑算法层，再看语境层**: 先 `algorithm_only=True` 得到可复现基线，再 `algorithm_only=False` 查看调校差异
2. **每次校准后重新跑所有引擎**: `v3 → v5 → v6 → backtest`
3. **永远不修改 `data/raw/` 下的文件**: 原始数据不可变，修正只发生在 `data/curated/`
4. **回测新版本**: 每次改动引擎后，运行 `python engine/backtest.py` 确认预测精度没有退化

---

## 附录A: 断言可校验性

每个断言预测都有3个可校验条件。到目标时间后，逐一检查：

| 断言 | 校验条件1 | 校验条件2 | 校验条件3 |
|:----|:---------|:---------|:---------|
| 1 Agent爆发 | DAU>1亿Agent产品 | OSWorld>80% | Agent错误率<15% |
| 2 成本降50% | 成本<$0.08/1M | ≥3家<$0.10 | token付费新模式 |
| 3 自改进成熟 | intel×prog交互效应>8% | 自我改进商用产品 | SWE-bench 100%>1月 |
| 4 CN-US缩至15% | CN-US parity≥85% | 纯能力差距≤15% | 中国模型基准第一 |
| 5 普及率65% | 全球采用率>65% | 企业采用率>90% | 日常使用AI>50% |
| 6 能源议题 | ≥3国出台AI能源法规 | 企业因能源调整战略 | 选址偏向绿电 |
| E1 世界模型商用 | 世界模型得分>700 | 物理模拟商用产品 | 因果推理基准>80% |
| E2 通用物理操作 | 物理交互得分>700 | 通用机器人商用 | Sim-to-Real gap<20% |
| E4 因果推理超越 | 抽象因果推理>90% | 干预推理(do-calculus) | 反事实推理商用 |
