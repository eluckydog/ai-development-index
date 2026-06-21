"""
AIDI v3 — 多维AI能力指数
==============================
AIC = 综合能力 (向量+交互效应)
AIDI = 发展速度 (向量模长增量)

能力空间: 6个维度 + 跨维度交互乘数
"""

import json, math
from datetime import datetime
from pathlib import Path

# ── 六维能力定义 ─────────────────────────────────────────────
DIMS = [
    "intelligence",  # LLM推理/知识/编程 - "大脑"
    "multimodal",    # 看/听/说/生成 - "感官"
    "agent",         # 工具/规划/自主/编排 - "手脚"
    "programming",   # 代码生成/自我改进 - "自我复制"
    "knowledge",     # RAG/向量库/记忆/检索 - "图书馆"
    "ecosystem",     # API/成本/框架/开发者生态 - "基础设施"
]

DIM_NAMES_ZH = {
    "intelligence": "语言智力",
    "multimodal": "多模态感知",
    "agent": "智能体行动力",
    "programming": "编程自改进",
    "knowledge": "知识系统",
    "ecosystem": "生态基础设施",
}

# ── 交互效应系数 ─────────────────────────────────────────────
# synergy[i][j] = 维度i和j的交互强度 (0-1)
# 语义: "当两个维度都强时，综合效果超过单维度之和"
# 数学形式: 交互项 = synergy × min(vi,vj)/1000 × (vi×vj/1e6)
SYNERGY = {
    ("intelligence", "agent"): 0.35,       # 大脑+手脚 = 能自主做事
    ("intelligence", "multimodal"): 0.20,  # 大脑+感官 = 理解真实世界
    ("intelligence", "programming"): 0.30, # 大脑+编程 = AI造AI
    ("intelligence", "knowledge"): 0.25,   # 大脑+图书馆 = 专家系统
    ("agent", "programming"): 0.25,        # 手脚+编程 = 自我改进工具
    ("agent", "ecosystem"): 0.20,          # 手脚+基础设施 = 普惠AI
    ("agent", "knowledge"): 0.20,          # 手脚+知识 = 有记忆的行动
    ("multimodal", "programming"): 0.15,   # 感官+编程 = 视觉编程
    ("multimodal", "agent"): 0.20,         # 感官+手脚 = 机器人
    ("knowledge", "ecosystem"): 0.15,      # 知识+生态 = 企业级知识库
    ("programming", "ecosystem"): 0.15,    # 编程+生态 = 应用生态
}

def calc_dim_aic(dim_scores):
    """计算维度基础AIC = 加权和 (权重平等)"""
    return sum(dim_scores.values()) / len(dim_scores)

def calc_synergy(dim_scores):
    """计算跨维度交互效应 (0~500加成)"""
    total_synergy = 0
    for (d1, d2), s in SYNERGY.items():
        v1 = dim_scores.get(d1, 0)
        v2 = dim_scores.get(d2, 0)
        if v1 > 0 and v2 > 0:
            # 交互项: 受弱维度约束 × 乘积效应
            weak = min(v1, v2) / 1000.0
            product = (v1 * v2) / 1_000_000.0
            synergy_term = s * weak * product * 500
            total_synergy += synergy_term
    return total_synergy

def calc_aic_v3(dim_scores):
    """AIC = 基础能力 + 交互效应"""
    base = calc_dim_aic(dim_scores)
    synergy = calc_synergy(dim_scores)
    return round(base + synergy)

def calc_aidi_v3(prev_aic, curr_aic):
    """AIDI = AIC增量"""
    return curr_aic - prev_aic

# ── 逐期历史数据: 六维评分 (0-1000) ──────────────────────────
# 基于关键里程碑估算, 每条代表该半月的综合能力水平
# intelligence分 = LLM能力
# multimodal分 = 图像/视频/音频/语音识别/生成
# agent分 = 工具调用/推理规划/自主Agent/多Agent编排
# programming分 = 代码生成/代码理解/自我改进
# knowledge分 = RAG/向量库/记忆/检索/知识图谱
# ecosystem分 = API可用性/成本/框架成熟度/开发者社区/普及度

PERIOD_SIX_DIMS = {
    # ═══ 2022 ═══
    "2022-12-01": {
        "intelligence": 100, "multimodal": 10, "agent": 5,
        "programming": 80, "knowledge": 10, "ecosystem": 30,
        "note": "ChatGPT上线, 编程有Copilot/Codex, 其他刚起步"
    },
    "2022-12-16": {
        "intelligence": 120, "multimodal": 10, "agent": 5,
        "programming": 85, "knowledge": 10, "ecosystem": 35,
        "note": "ChatGPT爆发, 微软百亿投资"
    },
    # ═══ 2023 ═══
    "2023-01-01": {
        "intelligence": 130, "multimodal": 12, "agent": 8,
        "programming": 90, "knowledge": 15, "ecosystem": 40,
    },
    "2023-01-16": {
        "intelligence": 140, "multimodal": 12, "agent": 8,
        "programming": 90, "knowledge": 15, "ecosystem": 45,
    },
    "2023-02-01": {
        "intelligence": 150, "multimodal": 15, "agent": 10,
        "programming": 95, "knowledge": 15, "ecosystem": 45,
    },
    "2023-02-16": {
        "intelligence": 200, "multimodal": 15, "agent": 10,
        "programming": 100, "knowledge": 20, "ecosystem": 50,
        "note": "LLaMA泄露, 开源AI萌芽"
    },
    "2023-03-01": {
        "intelligence": 400, "multimodal": 200, "agent": 20,
        "programming": 200, "knowledge": 30, "ecosystem": 60,
        "note": "GPT-4多模态+编程飞跃, Claude 1同日发布"
    },
    "2023-03-16": {
        "intelligence": 420, "multimodal": 220, "agent": 25,
        "programming": 220, "knowledge": 35, "ecosystem": 65,
    },
    "2023-04-01": {
        "intelligence": 430, "multimodal": 230, "agent": 30,
        "programming": 230, "knowledge": 40, "ecosystem": 70,
    },
    "2023-04-16": {
        "intelligence": 440, "multimodal": 240, "agent": 35,
        "programming": 240, "knowledge": 45, "ecosystem": 75,
    },
    "2023-05-01": {
        "intelligence": 450, "multimodal": 250, "agent": 40,
        "programming": 250, "knowledge": 50, "ecosystem": 80,
    },
    "2023-05-16": {
        "intelligence": 460, "multimodal": 260, "agent": 45,
        "programming": 255, "knowledge": 55, "ecosystem": 85,
    },
    "2023-06-01": {
        "intelligence": 470, "multimodal": 270, "agent": 50,
        "programming": 260, "knowledge": 60, "ecosystem": 90,
    },
    "2023-06-16": {
        "intelligence": 480, "multimodal": 280, "agent": 55,
        "programming": 270, "knowledge": 65, "ecosystem": 95,
    },
    "2023-07-01": {
        "intelligence": 520, "multimodal": 290, "agent": 60,
        "programming": 300, "knowledge": 70, "ecosystem": 100,
        "note": "Claude 2 (100K上下文), LLaMA 2开源可商用"
    },
    "2023-07-16": {
        "intelligence": 530, "multimodal": 300, "agent": 65,
        "programming": 310, "knowledge": 75, "ecosystem": 105,
    },
    "2023-08-01": {
        "intelligence": 550, "multimodal": 310, "agent": 70,
        "programming": 350, "knowledge": 80, "ecosystem": 110,
        "note": "Code Llama发布, 代码能力专精"
    },
    "2023-08-16": {
        "intelligence": 560, "multimodal": 320, "agent": 75,
        "programming": 360, "knowledge": 85, "ecosystem": 115,
    },
    "2023-09-01": {
        "intelligence": 570, "multimodal": 400, "agent": 80,
        "programming": 370, "knowledge": 90, "ecosystem": 120,
        "note": "GPT-4V多模态看懂图片"
    },
    "2023-09-16": {
        "intelligence": 580, "multimodal": 410, "agent": 85,
        "programming": 380, "knowledge": 95, "ecosystem": 125,
    },
    "2023-10-01": {
        "intelligence": 590, "multimodal": 420, "agent": 90,
        "programming": 390, "knowledge": 100, "ecosystem": 130,
    },
    "2023-10-16": {
        "intelligence": 600, "multimodal": 430, "agent": 95,
        "programming": 400, "knowledge": 105, "ecosystem": 135,
    },
    "2023-11-01": {
        "intelligence": 620, "multimodal": 440, "agent": 100,
        "programming": 420, "knowledge": 110, "ecosystem": 150,
        "note": "Grok 1实时知识, Claude 2.1 (200K上下文)"
    },
    "2023-11-16": {
        "intelligence": 630, "multimodal": 450, "agent": 105,
        "programming": 430, "knowledge": 120, "ecosystem": 155,
        "note": "DeepSeek Coder崛起, 开源追赶"
    },
    "2023-12-01": {
        "intelligence": 650, "multimodal": 500, "agent": 110,
        "programming": 440, "knowledge": 130, "ecosystem": 160,
        "note": "Gemini 1.0原生多模态, MMLU 90%"
    },
    "2023-12-16": {
        "intelligence": 660, "multimodal": 510, "agent": 115,
        "programming": 450, "knowledge": 140, "ecosystem": 165,
    },
    # ═══ 2024 ═══
    "2024-01-01": {
        "intelligence": 670, "multimodal": 520, "agent": 120,
        "programming": 460, "knowledge": 150, "ecosystem": 170,
    },
    "2024-01-16": {
        "intelligence": 690, "multimodal": 530, "agent": 125,
        "programming": 500, "knowledge": 160, "ecosystem": 180,
        "note": "Code Llama 70B, DeepSeek MoE创新"
    },
    "2024-02-01": {
        "intelligence": 720, "multimodal": 550, "agent": 130,
        "programming": 510, "knowledge": 170, "ecosystem": 190,
        "note": "Gemini 1.0 Ultra, Gemini 1.5 Pro百万token"
    },
    "2024-02-16": {
        "intelligence": 730, "multimodal": 560, "agent": 135,
        "programming": 520, "knowledge": 180, "ecosystem": 195,
    },
    "2024-03-01": {
        "intelligence": 780, "multimodal": 580, "agent": 150,
        "programming": 550, "knowledge": 200, "ecosystem": 200,
        "note": "Claude 3 Opus全面超越GPT-4, 推理+编程同时提升"
    },
    "2024-03-16": {
        "intelligence": 790, "multimodal": 590, "agent": 160,
        "programming": 560, "knowledge": 210, "ecosystem": 210,
    },
    "2024-04-01": {
        "intelligence": 800, "multimodal": 600, "agent": 170,
        "programming": 600, "knowledge": 220, "ecosystem": 220,
        "note": "LLaMA 3开源, DeepSeek V2性价比"
    },
    "2024-04-16": {
        "intelligence": 810, "multimodal": 610, "agent": 180,
        "programming": 610, "knowledge": 230, "ecosystem": 230,
        "note": "DeepSeek V2 MoE 236B"
    },
    "2024-05-01": {
        "intelligence": 850, "multimodal": 750, "agent": 200,
        "programming": 620, "knowledge": 250, "ecosystem": 250,
        "note": "GPT-4o全程多模态实时语音, Agent雏形出现"
    },
    "2024-05-16": {
        "intelligence": 860, "multimodal": 760, "agent": 210,
        "programming": 630, "knowledge": 260, "ecosystem": 260,
    },
    "2024-06-01": {
        "intelligence": 880, "multimodal": 770, "agent": 220,
        "programming": 680, "knowledge": 270, "ecosystem": 270,
        "note": "Claude 3.5 Sonnet编程领先, RAG技术成熟"
    },
    "2024-06-16": {
        "intelligence": 890, "multimodal": 780, "agent": 230,
        "programming": 690, "knowledge": 280, "ecosystem": 280,
    },
    "2024-07-01": {
        "intelligence": 900, "multimodal": 790, "agent": 250,
        "programming": 720, "knowledge": 300, "ecosystem": 300,
        "note": "GPT-4o mini普及, LLaMA 3.1 405B最大开源, Agent框架兴起"
    },
    "2024-07-16": {
        "intelligence": 910, "multimodal": 800, "agent": 260,
        "programming": 730, "knowledge": 310, "ecosystem": 310,
    },
    "2024-08-01": {
        "intelligence": 920, "multimodal": 810, "agent": 270,
        "programming": 740, "knowledge": 320, "ecosystem": 320,
        "note": "Grok 2多模态"
    },
    "2024-08-16": {
        "intelligence": 930, "multimodal": 820, "agent": 280,
        "programming": 750, "knowledge": 330, "ecosystem": 330,
    },
    "2024-09-01": {
        "intelligence": 950, "multimodal": 830, "agent": 400,
        "programming": 760, "knowledge": 350, "ecosystem": 340,
        "note": "o1推理模型突破! Agent能力跃升(学会了推理)"
    },
    "2024-09-16": {
        "intelligence": 960, "multimodal": 840, "agent": 420,
        "programming": 770, "knowledge": 360, "ecosystem": 350,
        "note": "Gemini 1.5 Pro-002"
    },
    "2024-10-01": {
        "intelligence": 970, "multimodal": 850, "agent": 430,
        "programming": 780, "knowledge": 370, "ecosystem": 360,
    },
    "2024-10-16": {
        "intelligence": 980, "multimodal": 860, "agent": 440,
        "programming": 790, "knowledge": 380, "ecosystem": 370,
        "note": "Claude 3.5 Haiku"
    },
    "2024-11-01": {
        "intelligence": 990, "multimodal": 870, "agent": 450,
        "programming": 800, "knowledge": 400, "ecosystem": 380,
    },
    "2024-11-16": {
        "intelligence": 1000, "multimodal": 880, "agent": 460,
        "programming": 810, "knowledge": 420, "ecosystem": 390,
        "note": "DeepSeek R1推理预览"
    },
    "2024-12-01": {
        "intelligence": 1050, "multimodal": 890, "agent": 480,
        "programming": 830, "knowledge": 450, "ecosystem": 500,
        "note": "DeepSeek V3开源逼近GPT-4o, 成本骤降! o1正式版推理能力"
    },
    "2024-12-16": {
        "intelligence": 1060, "multimodal": 900, "agent": 490,
        "programming": 840, "knowledge": 460, "ecosystem": 520,
    },
    # ═══ 2025 ═══
    "2025-01-01": {
        "intelligence": 1100, "multimodal": 910, "agent": 500,
        "programming": 850, "knowledge": 480, "ecosystem": 530,
        "note": "o3-mini推理, ARC-AGI突破"
    },
    "2025-01-16": {
        "intelligence": 1120, "multimodal": 920, "agent": 520,
        "programming": 860, "knowledge": 500, "ecosystem": 550,
        "note": "DeepSeek R1推理模型震惊全球"
    },
    "2025-02-01": {
        "intelligence": 1150, "multimodal": 930, "agent": 530,
        "programming": 870, "knowledge": 510, "ecosystem": 560,
        "note": "Grok 3, GPT-4.5最后非推理旗舰"
    },
    "2025-02-16": {
        "intelligence": 1160, "multimodal": 940, "agent": 540,
        "programming": 880, "knowledge": 520, "ecosystem": 570,
    },
    "2025-03-01": {
        "intelligence": 1200, "multimodal": 950, "agent": 550,
        "programming": 900, "knowledge": 550, "ecosystem": 580,
        "note": "Claude 3.7 Sonnet编程突破, Gemini 2.5 Pro百万token推理"
    },
    "2025-03-16": {
        "intelligence": 1210, "multimodal": 960, "agent": 560,
        "programming": 910, "knowledge": 560, "ecosystem": 590,
        "note": "DeepSeek V3-0324升级"
    },
    "2025-04-01": {
        "intelligence": 1250, "multimodal": 970, "agent": 580,
        "programming": 920, "knowledge": 570, "ecosystem": 600,
        "note": "LLaMA 4开源MoE, o3正式发布"
    },
    "2025-04-16": {
        "intelligence": 1260, "multimodal": 980, "agent": 590,
        "programming": 930, "knowledge": 580, "ecosystem": 610,
        "note": "Gemini 2.5 Flash"
    },
    "2025-05-01": {
        "intelligence": 1300, "multimodal": 990, "agent": 650,
        "programming": 950, "knowledge": 600, "ecosystem": 620,
        "note": "Claude Opus 4+Sonnet 4深度推理/编程领先, Agent框架成熟"
    },
    "2025-05-16": {
        "intelligence": 1310, "multimodal": 1000, "agent": 660,
        "programming": 960, "knowledge": 610, "ecosystem": 630,
    },
    "2025-06-01": {
        "intelligence": 1330, "multimodal": 1010, "agent": 680,
        "programming": 970, "knowledge": 630, "ecosystem": 650,
        "note": "o3-pro专业推理, DeepSeek R1升级"
    },
    "2025-06-16": {
        "intelligence": 1340, "multimodal": 1020, "agent": 690,
        "programming": 980, "knowledge": 640, "ecosystem": 660,
    },
    "2025-07-01": {
        "intelligence": 1350, "multimodal": 1030, "agent": 700,
        "programming": 990, "knowledge": 650, "ecosystem": 670,
    },
    "2025-07-16": {
        "intelligence": 1360, "multimodal": 1040, "agent": 710,
        "programming": 1000, "knowledge": 660, "ecosystem": 680,
    },
    "2025-08-01": {
        "intelligence": 1550, "multimodal": 1050, "agent": 800,
        "programming": 1200, "knowledge": 700, "ecosystem": 700,
        "note": "GPT-5发布! 百万token, Agent能力跃升, OpenAI首次开源"
    },
    "2025-08-16": {
        "intelligence": 1580, "multimodal": 1060, "agent": 810,
        "programming": 1210, "knowledge": 720, "ecosystem": 720,
        "note": "DeepSeek V3.1"
    },
    "2025-09-01": {
        "intelligence": 1600, "multimodal": 1070, "agent": 820,
        "programming": 1250, "knowledge": 740, "ecosystem": 740,
        "note": "GPT-5 Codex编程专精, Agent全面商用"
    },
    "2025-09-16": {
        "intelligence": 1620, "multimodal": 1080, "agent": 830,
        "programming": 1260, "knowledge": 760, "ecosystem": 760,
        "note": "Claude Sonnet 4.5, DeepSeek V3.2"
    },
    "2025-10-01": {
        "intelligence": 1640, "multimodal": 1090, "agent": 840,
        "programming": 1280, "knowledge": 780, "ecosystem": 780,
        "note": "Claude Haiku 4.5"
    },
    "2025-10-16": {
        "intelligence": 1650, "multimodal": 1100, "agent": 850,
        "programming": 1290, "knowledge": 800, "ecosystem": 800,
    },
    "2025-11-01": {
        "intelligence": 1750, "multimodal": 1120, "agent": 880,
        "programming": 1350, "knowledge": 850, "ecosystem": 850,
        "note": "GPT-5.1升级, Gemini 3.0 Pro, Claude Opus 4.5科学推理领先"
    },
    "2025-11-16": {
        "intelligence": 1760, "multimodal": 1130, "agent": 890,
        "programming": 1360, "knowledge": 860, "ecosystem": 860,
    },
    "2025-12-01": {
        "intelligence": 1800, "multimodal": 1150, "agent": 900,
        "programming": 1400, "knowledge": 900, "ecosystem": 880,
        "note": "GPT-5.2优化, Gemini 3.0 Flash"
    },
    "2025-12-16": {
        "intelligence": 1810, "multimodal": 1160, "agent": 910,
        "programming": 1410, "knowledge": 910, "ecosystem": 900,
    },
    # ═══ 2026 ═══
    "2026-01-01": {
        "intelligence": 1850, "multimodal": 1180, "agent": 930,
        "programming": 1450, "knowledge": 950, "ecosystem": 920,
        "note": "Gemini 2.0系列, o3-mini高算力版"
    },
    "2026-01-16": {
        "intelligence": 1860, "multimodal": 1190, "agent": 940,
        "programming": 1460, "knowledge": 960, "ecosystem": 940,
    },
    "2026-02-01": {
        "intelligence": 1950, "multimodal": 1210, "agent": 970,
        "programming": 1500, "knowledge": 980, "ecosystem": 970,
        "note": "Claude Opus 4.6, Grok 4, Gemini 3.1 Pro"
    },
    "2026-02-16": {
        "intelligence": 2000, "multimodal": 1220, "agent": 980,
        "programming": 1550, "knowledge": 1000, "ecosystem": 1000,
        "note": "GPT-5.3-Codex编程, Gemini 3.1 Flash"
    },
    "2026-03-01": {
        "intelligence": 2300, "multimodal": 1250, "agent": 1100,
        "programming": 1700, "knowledge": 1050, "ecosystem": 1100,
        "note": "GPT-5.4, Capybara泄露第四档, 三模型密集发布"
    },
    "2026-03-16": {
        "intelligence": 2350, "multimodal": 1260, "agent": 1120,
        "programming": 1750, "knowledge": 1080, "ecosystem": 1150,
        "note": "Muse Spark Meta新系列"
    },
    "2026-04-01": {
        "intelligence": 2500, "multimodal": 1280, "agent": 1300,
        "programming": 1900, "knowledge": 1150, "ecosystem": 1200,
        "note": "GPT-5.5发布! DeepSeek V4-Pro/Flash开源! SWE-Bench 83.7%, Agent进入新纪元"
    },
    "2026-04-16": {
        "intelligence": 2520, "multimodal": 1290, "agent": 1320,
        "programming": 1920, "knowledge": 1180, "ecosystem": 1250,
    },
    "2026-05-01": {
        "intelligence": 2600, "multimodal": 1400, "agent": 1450,
        "programming": 2000, "knowledge": 1250, "ecosystem": 1300,
        "note": "Gemini Omni全模态, Claude Opus 4.8科学推理登顶 (HLE 49.8%)"
    },
    "2026-05-16": {
        "intelligence": 2650, "multimodal": 1420, "agent": 1500,
        "programming": 2050, "knowledge": 1300, "ecosystem": 1350,
        "note": "Claude Fable 5发布, SWE-Bench 80.3%"
    },
    "2026-06-01": {
        "intelligence": 2700, "multimodal": 1450, "agent": 1550,
        "programming": 2100, "knowledge": 1350, "ecosystem": 1400,
        "note": "最新数据 "
    },
    "2026-06-16": {
        "intelligence": 2750, "multimodal": 1470, "agent": 1580,
        "programming": 2150, "knowledge": 1380, "ecosystem": 1450,
        "note": "当前半月 (预估)"
    },
}


def build_timeseries_v3():
    """构建完整时间序列 (含AIC/AIDI计算)"""
    periods = []
    prev_aic = None

    sorted_keys = sorted(PERIOD_SIX_DIMS.keys())
    
    for key in sorted_keys:
        raw = PERIOD_SIX_DIMS[key]
        dims = {k: v for k, v in raw.items() if k in DIMS}
        aic = calc_aic_v3(dims)
        
        if prev_aic is not None:
            aidi = calc_aidi_v3(prev_aic, aic)
        else:
            aidi = 100  # 基线速度 (用户设定)
        
        periods.append({
            "date": key,
            "aic": aic,
            "aidi": aidi,
            "dimensions": dims.copy(),
            "note": dims.get("note", ""),
        })
        prev_aic = aic

    return periods


def run_full_v3():
    """全量分析"""
    periods = build_timeseries_v3()
    base = periods[0]
    curr = periods[-1]

    print("=" * 65)
    print("AIDI v3 — 六维交互指数")
    print("=" * 65)
    print(f"基线: {base['date']}  AIC={base['aic']}  AIDI={base['aidi']}")
    print(f"当前: {curr['date']}  AIC={curr['aic']}  AIDI={curr['aidi']:+}")
    print(f"增长: +{curr['aic'] - base['aic']}点")
    print()
    print(f"{'日期':<14} {'AIC':>6} {'AIDI':>6} {'智力':>5} {'感官':>5} {'行动':>5} {'编程':>5} {'知识':>5} {'生态':>5}")
    print("-" * 65)

    # 关键期 (锚点或AIDI波动期)
    for p in periods:
        dims = p['dimensions']
        note = p.get('note', '')
        if note or p['aidi'] > 50 or abs(p['aidi']) < 15:
            i = dims.get('intelligence', 0)
            m = dims.get('multimodal', 0)
            a = dims.get('agent', 0)
            c = dims.get('programming', 0)
            k = dims.get('knowledge', 0)
            e = dims.get('ecosystem', 0)
            print(f"{p['date']:<14} {p['aic']:>6} {p['aidi']:>+6} {i:>5} {m:>5} {a:>5} {c:>5} {k:>5} {e:>5}")

    # 交互效应分析
    print()
    print("交互效应分析 (当前):")
    cur_dims = curr['dimensions']
    base_aic = calc_dim_aic(cur_dims)
    synergy = calc_synergy(cur_dims)
    print(f"  基础能力(六维平均): {base_aic:.0f}")
    print(f"  交互效应加成: +{synergy:.0f}")
    print(f"  AIC综合: {curr['aic']}")
    print(f"  交互占比: {synergy / curr['aic'] * 100:.1f}%")

    # 保存报告
    report = {
        "baseline": {"date": base["date"], "aic": base["aic"], "aidi": base["aidi"]},
        "current": {"date": curr["date"], "aic": curr["aic"], "aidi": curr["aidi"]},
        "aic_growth": curr["aic"] - base["aic"],
        "aic_growth_pct": round((curr["aic"] / base["aic"] - 1) * 100, 1),
        "current_dimensions": cur_dims,
        "base_aic": round(base_aic),
        "synergy_bonus": round(synergy),
        "synergy_pct": round(synergy / curr["aic"] * 100, 1),
        "total_periods": len(periods),
    }

    report_path = "reports/aidi_v3_report.json"
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存: {report_path}")

    return report


if __name__ == "__main__":
    run_full_v3()
