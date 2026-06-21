# AI Development Index (AIDI)

从2022年12月GPT发布至今的全球AI发展定量追踪系统。

**基线**: AIC=1000, AIDI=100km/h (2022-12-01, GPT发布日)  
**更新频率**: 每半月 (1日/16日)  
**定价**: 完全开源, 采集零成本 (纯免费API)

## 核心指标

| 指标 | 含义 | 物理类比 | 当前值 (2026-06) |
|:----|------|---------|:---------------:|
| **AIC** | AI能力总量 | 位置/里程 | 4900 (较基线+3900) |
| **AIDI** | AI发展速度 | 车速 km/h | 0~+59/半月 |
| **CN-US Parity** | 中美综合能力比 | % | 78% |
| **校验F** | 预测精度 | 0-1 | 0.914 (高精度) |

## 设计原理

### AIC (AI Capability Index)
- 基于国际公认基准加权: Arena Elo(35%) + MMLU(20%) + GPQA(20%) + SWE-bench(15%) + MATH(10%)
- 锚定24个关键模型发布事件, 插值生成86期时间序列
- 基线1000=GPT-4发布时的全球AI能力水平

### AIDI (AI Development Index)
- AIC的每半月增量 = 发展速度
- 爆发期可达+129/半月 (2025.01 o3发布)
- 平台期接近+0 (当前)

### 中美能力对比
- **综合parity** = 纯能力×0.5 + 性价比×0.25 + 开源×0.15 + 创新×0.1
- 纯能力基于国际基准: 中国TOP约为美国96%
- 当前综合: 中国=美国78%

### 预测与校验
- 贝叶斯预测: 基于历史趋势+波动性, 含95%置信区间
- 校验函数F: 回头看预测精度, 含MAPE/区间命中率/校准度
- 每期自动积累经验, 修正预测模型

## 仓库结构

```
├── data/
│   ├── raw/              ← 原始搜索数据 (不可修改)
│   ├── curated/          ← 加工后AIDI/AIC时间序列
│   └── benchmarks/       ← 外部基准测评
├── engine/
│   ├── aidicore.py       ← 核心引擎 (AIC/AIDI/parity/预测/校验)
│   ├── gen_periods.py    ← 86期日期生成器
│   └── first_run.py      ← 首次基线确认脚本
├── skills/aidi-search/   ← 搜索方法论
├── .github/workflows/
│   ├── aidi-collect.yml  ← 单期数据采集
│   └── aidi-backfill.yml ← 86期批量回填
└── reports/              ← 每期分析报告
```

## 使用

```bash
# 本地运行完整分析
python engine/aidicore.py

# 查看5条关键基线 (2022/2024/2025/2026基线+当前)
python engine/first_run.py

# 列出所有86期
python engine/gen_periods.py
```

## 数据来源 (零成本)

| 维度 | 来源 | 成本 |
|:----|------|:----:|
| Arena Elo | lmarena.ai (公开leaderboard) | 免费 |
| 模型基准M/MLU/GPQA | llm-stats.com, benchlm.ai | 免费 |
| GitHub热榜 | gh api | 免费 |
| RSS新闻 | feedparser | 免费 |
| 金融宏观 | AKShare/自有管线 | 免费 |
| arXiv论文 | arXiv API | 免费 |

## 验证

引擎通过三阶段检查:
- ✅ S1: 文件完整性 (7/7)
- ✅ S2: 内容正确性 (YAML/Python/JSON全部通过)
- ✅ S3: 可运行性 (workflow测试通过)
