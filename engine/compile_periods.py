"""
AIDI Period Data Compiler v2
基于AI Release Tracker真实数据 + WebSearch 补充，逐期填充每个半月的模型发布/基准/事件
"""
import json, os
from datetime import datetime, timedelta

# ── 真实数据：从AI Release Tracker获取 + WebSearch补充 ──
# 来源: aireleasetracker.com (OpenAI/Anthropic/Google/DeepSeek/Meta/xAI)
# 按周期分组 (period_name -> list of events)

PERIOD_DATA = {
    # ═══════════════════════════════════════════════════
    # 2022
    # ═══════════════════════════════════════════════════
    "2022-12-01": {
        "period_name": "基线期",
        "events": [
            ("OpenAI", "GPT-3.5 + ChatGPT", "2022-11-30", "ChatGPT上线, 首月1亿用户. MMLU=70.0, HumanEval=48.1. AI发展基线."),
        ],
        "ai_news": "ChatGPT发布引爆全球AI热潮, OpenAI估值290亿美元. Stable Diffusion 2.0发布.",
        "benchmarks": {"MMLU": 70.0, "HumanEval": 48.1},
        "model_score": 12, "hardware_score": 5, "algorithm_score": 10,
        "business_score": 15, "adoption_score": 20, "opensource_score": 5,
    },
    # ═══════════════════════════════════════════════════
    # 2023
    # ═══════════════════════════════════════════════════
    "2023-01-01": {
        "period_name": "ChatGPT发酵期",
        "events": [
            ("Microsoft", "投资OpenAI", "2023-01-23", "微软追加100亿美元投资, 整合ChatGPT到Bing/Office"),
        ],
        "ai_news": "ChatGPT月活1亿破纪录, 微软100亿投资OpenAI. Google发布Bard应对.",
        "benchmarks": {},
        "model_score": 15, "hardware_score": 5, "algorithm_score": 12,
        "business_score": 25, "adoption_score": 25, "opensource_score": 8,
    },
    "2023-01-16": {
        "events": [("Meta", "LLaMA泄露", "2023-03-03", "LLaMA模型权重泄露, 引发开源AI爆发")],
        "benchmarks": {},
    },
    "2023-02-01": {
        "events": [],
        "benchmarks": {},
    },
    "2023-02-16": {
        "events": [("Meta", "LLaMA 1", "2023-02-24", "Meta发布LLaMA, 开源大语言模型")],
        "benchmarks": {},
    },
    "2023-03-01": {
        "period_name": "GPT-4发布期",
        "events": [
            ("OpenAI", "GPT-4", "2023-03-14", "GPT-4发布. MMLU=86.4%, HumanEval=67.0%, 多模态. 里程碑时刻."),
            ("Anthropic", "Claude 1", "2023-03-14", "同日发布Claude 1."),
            ("Google", "Bard", "2023-03-21", "Google发布Bard聊天机器人."),
        ],
        "ai_news": "GPT-4发布, 支持多模态输入. Anthropic同日发布Claude 1. Google Bard匆忙上线翻车.",
        "benchmarks": {"MMLU": 86.4, "HumanEval": 67.0},
        "model_score": 30, "hardware_score": 10, "algorithm_score": 25,
        "business_score": 30, "adoption_score": 30, "opensource_score": 12,
    },
    "2023-03-16": {
        "events": [],
        "benchmarks": {},
    },
    "2023-04-01": {"events": [], "benchmarks": {}},
    "2023-04-16": {"events": [], "benchmarks": {}},
    "2023-05-01": {"events": [], "benchmarks": {}},
    "2023-05-16": {"events": [], "benchmarks": {}},
    "2023-06-01": {"events": [], "benchmarks": {}},
    "2023-06-16": {"events": [], "benchmarks": {}},
    "2023-07-01": {
        "events": [
            ("Anthropic", "Claude 2", "2023-07-11", "Claude 2发布, 100K token上下文, 编程能力提升"),
            ("Meta", "LLaMA 2", "2023-07-18", "LLaMA 2开源, 可商用"),
        ],
        "benchmarks": {},
    },
    "2023-07-16": {"events": [], "benchmarks": {}},
    "2023-08-01": {
        "events": [
            ("Meta", "Code Llama", "2023-08-24", "代码生成专用LLM"),
        ],
        "benchmarks": {},
    },
    "2023-08-16": {"events": [], "benchmarks": {}},
    "2023-09-01": {"events": [], "benchmarks": {}},
    "2023-09-16": {"events": [], "benchmarks": {}},
    "2023-10-01": {"events": [], "benchmarks": {}},
    "2023-10-16": {"events": [], "benchmarks": {}},
    "2023-11-01": {
        "events": [
            ("xAI", "Grok 1", "2023-11-04", "Elon Musk xAI发布Grok, 实时知识"),
            ("Anthropic", "Claude 2.1", "2023-11-21", "200K token上下文窗口"),
        ],
        "benchmarks": {},
    },
    "2023-11-16": {
        "events": [
            ("DeepSeek", "DeepSeek Coder", "2023-11-02", "开源代码模型"),
            ("DeepSeek", "DeepSeek-LLM", "2023-11-29", "DeepSeek首个通用大模型"),
        ],
        "benchmarks": {},
    },
    "2023-12-01": {
        "period_name": "Gemini发布期",
        "events": [
            ("Google", "Gemini 1.0", "2023-12-06/13", "Gemini Nano+Pro发布. 首个多模态原生模型. MMLU=90.0%."),
        ],
        "benchmarks": {"MMLU": 90.0},
    },
    "2023-12-16": {"events": [], "benchmarks": {}},
    # ═══════════════════════════════════════════════════
    # 2024
    # ═══════════════════════════════════════════════════
    "2024-01-01": {"events": [], "benchmarks": {}},
    "2024-01-16": {
        "events": [
            ("Meta", "Code Llama 70B", "2024-01-29", "最大代码模型"),
            ("DeepSeek", "DeepSeek-MoE", "2024-01-09", "MoE架构探索"),
        ],
        "benchmarks": {},
    },
    "2024-02-01": {
        "events": [
            ("Google", "Gemini 1.0 Ultra", "2024-02-08", "谷歌最强模型"),
            ("Google", "Gemini 1.5 Pro", "2024-02-15", "百万token上下文!! 里程碑"),
        ],
        "benchmarks": {"MMLU": 90.0},
    },
    "2024-02-16": {"events": [], "benchmarks": {}},
    "2024-03-01": {
        "events": [
            ("Anthropic", "Claude 3 Opus/Sonnet/Haiku", "2024-03-04", "Claude 3全家桶, Opus全面超越GPT-4"),
        ],
        "benchmarks": {"MMLU": 86.8, "GPQA": 50.4},
    },
    "2024-03-16": {"events": [], "benchmarks": {}},
    "2024-04-01": {
        "events": [
            ("Meta", "LLaMA 3 8B/70B", "2024-04-18", "LLaMA 3开源, 大幅提升"),
            ("DeepSeek", "DeepSeek-Math", "2024-04-03", "数学专用模型"),
        ],
        "benchmarks": {},
    },
    "2024-04-16": {
        "events": [
            ("DeepSeek", "DeepSeek V2", "2024-05-01", "MoE 236B, 性价比领先"),
        ],
        "benchmarks": {},
    },
    "2024-05-01": {
        "period_name": "GPT-4o发布期",
        "events": [
            ("OpenAI", "GPT-4o", "2024-05-13", "全程多模态, 实时语音, 免费. MMLU=88.7%"),
            ("Google", "Gemini 1.5 Flash", "2024-05-14", "轻量版Gemini"),
        ],
        "benchmarks": {"MMLU": 88.7},
    },
    "2024-05-16": {"events": [], "benchmarks": {}},
    "2024-06-01": {
        "events": [
            ("Anthropic", "Claude 3.5 Sonnet", "2024-06-20", "编程能力业界领先"),
            ("DeepSeek", "DeepSeek Coder V2", "2024-06-01", "开源代码模型升级"),
        ],
        "benchmarks": {},
    },
    "2024-06-16": {"events": [], "benchmarks": {}},
    "2024-07-01": {
        "events": [
            ("OpenAI", "GPT-4o mini", "2024-07-18", "廉价版GPT-4o"),
            ("Meta", "LLaMA 3.1 405B", "2024-07-23", "最大开源模型405B参数"),
        ],
        "benchmarks": {},
    },
    "2024-07-16": {"events": [], "benchmarks": {}},
    "2024-08-01": {
        "events": [
            ("xAI", "Grok 2", "2024-08", "多模态Grok"),
        ],
        "benchmarks": {},
    },
    "2024-08-16": {"events": [], "benchmarks": {}},
    "2024-09-01": {
        "events": [
            ("OpenAI", "o1-preview/o1-mini", "2024-09-12", "推理模型突破! o1是OpenAI首个推理模型"),
            ("DeepSeek", "DeepSeek V2.5", "2024-09-01", "模型升级"),
        ],
        "benchmarks": {"AIME": 55.0},
    },
    "2024-09-16": {
        "events": [
            ("Google", "Gemini 1.5 Pro-002/Flash-002", "2024-09-24", "Gemini升级版"),
        ],
        "benchmarks": {},
    },
    "2024-10-01": {"events": [], "benchmarks": {}},
    "2024-10-16": {
        "events": [
            ("Anthropic", "Claude 3.5 Haiku", "2024-10-22", "快速廉价模型"),
        ],
        "benchmarks": {},
    },
    "2024-11-01": {"events": [], "benchmarks": {}},
    "2024-11-16": {
        "events": [
            ("DeepSeek", "DeepSeek-R1-Lite", "2024-11-20", "推理模型预览"),
        ],
        "benchmarks": {},
    },
    "2024-12-01": {
        "period_name": "DeepSeek V3爆发期",
        "events": [
            ("DeepSeek", "DeepSeek V3", "2024-12-01", "MoE 671B, 开源逼近GPT-4o. 仅用$5.6M训练. 开源里程碑."),
            ("OpenAI", "o1正式版", "2024-12-05", "o1推理模型正式上线"),
            ("xAI", "Grok 2正式", "2024-12", "Grok 2全面上线"),
        ],
        "benchmarks": {"MMLU": 88.5, "GPQA": 59.1, "HumanEval": 82.6},
    },
    "2024-12-16": {"events": [], "benchmarks": {}},
    # ═══════════════════════════════════════════════════
    # 2025
    # ═══════════════════════════════════════════════════
    "2025-01-01": {
        "events": [
            ("OpenAI", "o3-mini", "2025-01-31", "推理模型, ARC AGI突破"),
        ],
        "benchmarks": {},
    },
    "2025-01-16": {
        "events": [
            ("DeepSeek", "DeepSeek Chat (R1)", "2025-01-20", "R1推理模型发布, 全球震惊"),
        ],
        "benchmarks": {},
    },
    "2025-02-01": {
        "period_name": "Grok 3发布期",
        "events": [
            ("xAI", "Grok 3", "2025-02", "Grok 3发布, App Store榜首"),
            ("OpenAI", "GPT-4.5", "2025-02-27", "OpenAI最后一个非推理旗舰"),
        ],
        "benchmarks": {},
    },
    "2025-02-16": {"events": [], "benchmarks": {}},
    "2025-03-01": {
        "events": [
            ("Anthropic", "Claude 3.7 Sonnet", "2025-02-24", "编程能力再次突破"),
            ("DeepSeek", "DeepSeek V3-0324", "2025-03-24", "V3重大升级"),
            ("Google", "Gemini 2.5 Pro", "2025-03-25", "推理能力新高度, 1M token"),
        ],
        "benchmarks": {},
    },
    "2025-03-16": {"events": [], "benchmarks": {}},
    "2025-04-01": {
        "events": [
            ("Meta", "LLaMA 4 Scout/Maverick", "2025-04-05", "LLaMA 4开源, MoE架构"),
            ("OpenAI", "o3/o4-mini", "2025-04-16", "o3推理模型正式发布"),
        ],
        "benchmarks": {},
    },
    "2025-04-16": {
        "events": [
            ("Google", "Gemini 2.5 Flash", "2025-04-17", "快速推理模型"),
        ],
        "benchmarks": {},
    },
    "2025-05-01": {
        "period_name": "Claude Opus 4期",
        "events": [
            ("Anthropic", "Claude Opus 4 / Sonnet 4", "2025-05-22", "Claude 4系列发布, 深度推理领先"),
            ("OpenAI", "GPT-4.1 / 4.1 mini", "2025-05-14", "GPT-4.1系列, 编程优化"),
        ],
        "benchmarks": {},
    },
    "2025-05-16": {"events": [], "benchmarks": {}},
    "2025-06-01": {
        "events": [
            ("DeepSeek", "DeepSeek R1-0528", "2025-05-28", "R1升级版"),
            ("OpenAI", "o3-pro", "2025-06-10", "o3专业版"),
        ],
        "benchmarks": {},
    },
    "2025-06-16": {
        "events": [
            ("Google", "Gemini 2.5 Flash-Lite", "2025-06-17", "超低成本模型"),
        ],
        "benchmarks": {},
    },
    "2025-07-01": {"events": [], "benchmarks": {}},
    "2025-07-16": {"events": [], "benchmarks": {}},
    "2025-08-01": {
        "period_name": "GPT-5发布期",
        "events": [
            ("OpenAI", "GPT-5 / GPT-5 mini", "2025-08-07", "GPT-5发布! 百万token上下文. 里程碑."),
            ("OpenAI", "gpt-oss-20b/120b", "2025-08-05", "OpenAI首次开源模型"),
            ("Anthropic", "Claude Opus 4.1", "2025-08-05", "Opus 4增强版"),
        ],
        "benchmarks": {},
    },
    "2025-08-16": {
        "events": [
            ("DeepSeek", "DeepSeek V3.1", "2025-08-21", "V3重大升级版"),
        ],
        "benchmarks": {},
    },
    "2025-09-01": {
        "events": [
            ("OpenAI", "GPT-5 Codex", "2025-09-15", "代码专用模型"),
            ("DeepSeek", "DeepSeek V3.1 Terminus", "2025-09-22", "V3.1终极版"),
        ],
        "benchmarks": {},
    },
    "2025-09-16": {
        "events": [
            ("Anthropic", "Claude Sonnet 4.5", "2025-09-29", "Sonnet升级版"),
            ("DeepSeek", "DeepSeek V3.2 Exp", "2025-09-29", "V3.2实验版"),
        ],
        "benchmarks": {},
    },
    "2025-10-01": {
        "events": [
            ("Anthropic", "Claude Haiku 4.5", "2025-10-15", "Haiku升级版"),
        ],
        "benchmarks": {},
    },
    "2025-10-16": {"events": [], "benchmarks": {}},
    "2025-11-01": {
        "period_name": "Claude 4.5 + GPT-5.1期",
        "events": [
            ("OpenAI", "GPT-5.1", "2025-11-12", "GPT-5升级版"),
            ("OpenAI", "GPT-5.1 Codex-Max", "2025-11-19", "代码模型升级"),
            ("Google", "Gemini 3.0 Pro", "2025-11-18", "Gemini 3.0系列"),
            ("Anthropic", "Claude Opus 4.5", "2025-11-24", "科学推理领先"),
        ],
        "benchmarks": {},
    },
    "2025-11-16": {
        "events": [
            ("DeepSeek", "DeepSeek V3.2", "2025-12-01", "V3系列终版"),
        ],
        "benchmarks": {},
    },
    "2025-12-01": {
        "events": [
            ("OpenAI", "GPT-5.2", "2025-12-11", "GPT-5进一步优化"),
            ("Google", "Gemini 3.0 Flash", "2025-12-17", "快速模型系列"),
        ],
        "benchmarks": {},
    },
    "2025-12-16": {"events": [], "benchmarks": {}},
    # ═══════════════════════════════════════════════════
    # 2026
    # ═══════════════════════════════════════════════════
    "2026-01-01": {
        "events": [
            ("Google", "Gemini 2.0 Flash/Pro", "2026-01-30-02-05", "Gemini 2.0系列正式发布"),
            ("OpenAI", "o3-mini高算力版", "2026-01-31", "推理模型升级"),
        ],
        "benchmarks": {},
    },
    "2026-01-16": {
        "events": [], "benchmarks": {},
    },
    "2026-02-01": {
        "period_name": "Claude Opus 4.6 + Gemini 3.1期",
        "events": [
            ("Anthropic", "Claude Opus 4.6", "2026-02-05", "Opus升级"),
            ("xAI", "Grok 4发布", "2026-02", "xAI新一代模型"),
            ("Anthropic", "Claude Sonnet 4.6", "2026-02-17", "Sonnet升级"),
            ("Google", "Gemini 3.1 Pro", "2026-02-19", "Gemini 3.1系列"),
        ],
        "benchmarks": {},
    },
    "2026-02-16": {
        "events": [
            ("Google", "Gemini 3.1 Flash", "2026-02-26", "快速模型"),
            ("OpenAI", "GPT-5.3-Codex", "2026-02-05", "代码模型"),
            ("OpenAI", "GPT-5.3-Codex-Spark", "2026-02-12", "快速代码模型"),
        ],
        "benchmarks": {},
    },
    "2026-03-01": {
        "period_name": "多模型密集发布期",
        "events": [
            ("OpenAI", "GPT-5.3-Instant", "2026-03-03", "超快速GPT"),
            ("OpenAI", "GPT-5.4 / 5.4-Pro", "2026-03-05", "GPT-5.4重大升级"),
            ("Google", "Gemini 3.1 Flash-Lite", "2026-03-03", "超轻量模型"),
            ("Anthropic", "Claude Mythos/Capybara泄露", "2026-03-15", "Anthropic第四档模型泄露, 性能惊人"),
        ],
        "benchmarks": {},
    },
    "2026-03-16": {
        "events": [
            ("Meta", "Muse Spark", "2026-04-08", "Meta新模型系列"),
        ],
        "benchmarks": {},
    },
    "2026-04-01": {
        "period_name": "GPT-5.5 + DeepSeek V4期",
        "events": [
            ("Anthropic", "Claude Opus 4.7", "2026-04-16", "Opus升级"),
            ("OpenAI", "GPT-5.5 / 5.5-Pro", "2026-04-23", "GPT-5.5发布! SWE-Bench 58.6%"),
            ("DeepSeek", "DeepSeek V4-Pro/Flash", "2026-04-24", "V4正式发布, SWE-Bench 83.7%, 价格仅1/20"),
        ],
        "benchmarks": {},
    },
    "2026-04-16": {
        "events": [], "benchmarks": {},
    },
    "2026-05-01": {
        "period_name": "Claude Opus 4.8 + Gemini Omni期",
        "events": [
            ("Google", "Gemini Omni / 3.5 Flash", "2026-05-19", "Gemini全模态模型"),
            ("Anthropic", "Claude Opus 4.8", "2026-05-28", "科学推理登顶, Humanity Last Exam 49.8%"),
            ("xAI", "Grok Build 0.1", "2026-05", "编程Agent模型"),
        ],
        "benchmarks": {},
    },
    "2026-05-16": {
        "events": [
            ("Anthropic", "Claude Fable 5", "2026-06-09", "Anthropic最新旗舰, SWE-Bench 80.3%"),
        ],
        "benchmarks": {},
    },
    "2026-06-01": {
        "period_name": "当前期",
        "events": [
            ("Anthropic", "Claude Fable 5", "2026-06-09", "最新旗舰"),
        ],
        "benchmarks": {"SWE-Bench": 80.3, "HLE": 49.8},
    },
    "2026-06-16": {
        "events": [], "benchmarks": {},
    },
}

# ── 生成 _search.json ──
base_dir = "data/raw"
count = 0
for period_name, data in PERIOD_DATA.items():
    events = data.get("events", [])
    benchmarks = data.get("benchmarks", {})
    
    # Build event list
    event_list = []
    for e in events:
        org, model, date, desc = e if len(e) == 4 else (e[0], e[1], e[2], "")
        event_list.append({
            "org": org, "model": model, "date": date, "description": desc
        })
    
    # Determine model activity score
    n_models = len(events)
    model_activity = min(n_models * 15, 100)
    research_activity = data.get("model_score", max(n_models * 10, 20))
    
    entry = {
        "period": {"start": period_name, "name": period_name.split("-")[0]},
        "searched_at": "2026-06-21 (manual/AI Release Tracker)",
        "sources": ["aireleasetracker.com", "websearch"],
        "summary": {
            "model_activity": model_activity,
            "research_activity": research_activity,
            "hype_models": [e["model"][:20] for e in event_list[:5]],
            "period_name": data.get("period_name", ""),
            "ai_news": data.get("ai_news", ""),
        },
        "releases": event_list,
        "benchmarks_found": benchmarks,
    }
    
    out_dir = os.path.join(base_dir, period_name)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "_search.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    
    count += 1

print(f"Filled {count} periods with real release data")
print()
# Verify key periods
for p in sorted(PERIOD_DATA.keys()):
    e = PERIOD_DATA[p]["events"]
    if e:
        print(f"  {p}: {len(e)} events - {e[0][1]}")