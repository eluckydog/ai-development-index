"""
AIDI v4 — 发展速度引擎 (Development Velocity Engine)
========================================================
基于 Hermes 算法方法论: PSO权重优化 + HMM状态检测 + 卡尔曼平滑

与 v3 (能力维度引擎) 互补:
  v3 = AIC(有多强)  → 能力向量 + 交互效应
  v4 = DVI(有多快)  → 发展速度 + 驱动力分析

三模块:
  1. 发展速度指数 (DVI): 供应侧3维 + 需求侧1维
  2. 能力指数 (复用v3): 读取dim_scores.json
  3. 中美比较: 差距分解

算法依据:
  - PSO粒子群优化 → 权重因子设计 (替代拍脑袋)
  - 卡尔曼滤波 → 趋势平滑 (去除周期噪声)
  - HMM隐马尔可夫 → 发展阶段检测 (加速/平台/拐点)
  - 格兰杰因果 → 维度间驱动关系
"""

import json, math, random, copy
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent

# ══════════════════════════════════════════════════════════════
# 1. 发展速度维度 (Development Velocity Indicators)
# ══════════════════════════════════════════════════════════════
# 供应侧: 决定"能不能快速发展"
#   硬件发展: GPU算力/HBM/推理芯片/边缘AI
#   算法效率: 量化/蒸馏/MoE/加速 → 算力利用率
#   商业价格: API定价/训练成本/投资
# 需求侧: 决定"实际多快"
#   应用渗透: Agent/工具/企业采用/用户活跃

DVI_DIMS = ["hardware", "efficiency", "pricing", "adoption"]

DVI_NAMES_ZH = {
    "hardware": "硬件发展速度",
    "efficiency": "算法效率提升",
    "pricing": "成本下降速度",
    "adoption": "应用渗透速度",
}

# ── 逐期历史数据: 四维发展速度评分 (0-1000) ──
# 每条代表该半月的"发展速度"而非"能力水平"
# hardware: GPU FLOPs增长 / HBM代际 / 推理芯片进展
# efficiency: 每token算力需求下降 / 量化/蒸馏进展
# pricing: 每百万token价格下降率
# adoption: AI工具DAU增长 / 企业采用率提升

DVI_HISTORY = {
    "2022-12-01": {
        "hardware": 60,
        "efficiency": 40,
        "pricing": 30,
        "adoption": 20,
        "note": "ChatGPT爆发, 但硬件和价格还未跟上"
    },
    "2022-12-16": {
        "hardware": 55,
        "efficiency": 42,
        "pricing": 32,
        "adoption": 35,
        "note": "用户暴涨, 服务器承压"
    },
    "2023-01-01": {
        "hardware": 55,
        "efficiency": 43,
        "pricing": 35,
        "adoption": 45
    },
    "2023-01-16": {
        "hardware": 55,
        "efficiency": 43,
        "pricing": 38,
        "adoption": 55
    },
    "2023-02-01": {
        "hardware": 55,
        "efficiency": 44,
        "pricing": 38,
        "adoption": 60
    },
    "2023-02-16": {
        "hardware": 60,
        "efficiency": 50,
        "pricing": 40,
        "adoption": 65,
        "note": "LLaMA开源, 推理成本开始下降"
    },
    "2023-03-01": {
        "hardware": 80,
        "efficiency": 55,
        "pricing": 30,
        "adoption": 75,
        "note": "GPT-4发布, NVIDIA H100需求暴增"
    },
    "2023-03-16": {
        "hardware": 68,
        "efficiency": 58,
        "pricing": 45,
        "adoption": 80
    },
    "2023-04-01": {
        "hardware": 70,
        "efficiency": 60,
        "pricing": 48,
        "adoption": 85
    },
    "2023-04-16": {
        "hardware": 72,
        "efficiency": 62,
        "pricing": 50,
        "adoption": 88
    },
    "2023-05-01": {
        "hardware": 75,
        "efficiency": 65,
        "pricing": 52,
        "adoption": 90
    },
    "2023-05-16": {
        "hardware": 78,
        "efficiency": 65,
        "pricing": 55,
        "adoption": 92
    },
    "2023-06-01": {
        "hardware": 80,
        "efficiency": 68,
        "pricing": 55,
        "adoption": 95
    },
    "2023-06-16": {
        "hardware": 82,
        "efficiency": 70,
        "pricing": 58,
        "adoption": 100
    },
    "2023-07-01": {
        "hardware": 85,
        "efficiency": 72,
        "pricing": 60,
        "adoption": 110,
        "note": "Claude 2(100K上下文), LLaMA 2商用开源"
    },
    "2023-07-16": {
        "hardware": 88,
        "efficiency": 73,
        "pricing": 62,
        "adoption": 115
    },
    "2023-08-01": {
        "hardware": 90,
        "efficiency": 78,
        "pricing": 65,
        "adoption": 120,
        "note": "Code Llama, 推理优化框架涌现"
    },
    "2023-08-16": {
        "hardware": 92,
        "efficiency": 80,
        "pricing": 68,
        "adoption": 125
    },
    "2023-09-01": {
        "hardware": 95,
        "efficiency": 82,
        "pricing": 70,
        "adoption": 130,
        "note": "GPT-4V多模态, vLLM推理优化"
    },
    "2023-09-16": {
        "hardware": 98,
        "efficiency": 85,
        "pricing": 72,
        "adoption": 135
    },
    "2023-10-01": {
        "hardware": 100,
        "efficiency": 88,
        "pricing": 75,
        "adoption": 140
    },
    "2023-10-16": {
        "hardware": 105,
        "efficiency": 90,
        "pricing": 78,
        "adoption": 145
    },
    "2023-11-01": {
        "hardware": 110,
        "efficiency": 95,
        "pricing": 50,
        "adoption": 160,
        "note": "Grok 1, Claude 2.1 200K, 推理成本再降"
    },
    "2023-11-16": {
        "hardware": 115,
        "efficiency": 100,
        "pricing": 85,
        "adoption": 170,
        "note": "DeepSeek Coder, 开源效率竞赛开始"
    },
    "2023-12-01": {
        "hardware": 120,
        "efficiency": 110,
        "pricing": 90,
        "adoption": 180,
        "note": "Gemini 1.0原生多模态, HBM3量产"
    },
    "2023-12-16": {
        "hardware": 140,
        "efficiency": 115,
        "pricing": 95,
        "adoption": 190
    },
    "2024-01-01": {
        "hardware": 150,
        "efficiency": 120,
        "pricing": 100,
        "adoption": 200
    },
    "2024-01-16": {
        "hardware": 160,
        "efficiency": 130,
        "pricing": 110,
        "adoption": 220,
        "note": "DeepSeek MoE创新, 推理效率突破"
    },
    "2024-02-01": {
        "hardware": 180,
        "efficiency": 140,
        "pricing": 120,
        "adoption": 240,
        "note": "Gemini 1.5 Pro百万token, 长上下文推理成本骤降"
    },
    "2024-02-16": {
        "hardware": 190,
        "efficiency": 145,
        "pricing": 125,
        "adoption": 250
    },
    "2024-03-01": {
        "hardware": 200,
        "efficiency": 160,
        "pricing": 135,
        "adoption": 270,
        "note": "Claude 3 Opus全面超越GPT-4, HW加速"
    },
    "2024-03-16": {
        "hardware": 210,
        "efficiency": 165,
        "pricing": 140,
        "adoption": 280
    },
    "2024-04-01": {
        "hardware": 220,
        "efficiency": 180,
        "pricing": 150,
        "adoption": 300,
        "note": "LLaMA 3开源, Groq LPU推理加速"
    },
    "2024-04-16": {
        "hardware": 230,
        "efficiency": 190,
        "pricing": 160,
        "adoption": 320,
        "note": "DeepSeek V2 MoE 236B, 推理效率翻倍"
    },
    "2024-05-01": {
        "hardware": 280,
        "efficiency": 220,
        "pricing": 150,
        "adoption": 380,
        "note": "GPT-4o发布, API价格战开始! 实时语音推理优化"
    },
    "2024-05-16": {
        "hardware": 270,
        "efficiency": 230,
        "pricing": 210,
        "adoption": 400
    },
    "2024-06-01": {
        "hardware": 280,
        "efficiency": 260,
        "pricing": 220,
        "adoption": 430,
        "note": "Claude 3.5 Sonnet, RAG技术成熟, AI编程工具爆发"
    },
    "2024-06-16": {
        "hardware": 290,
        "efficiency": 270,
        "pricing": 230,
        "adoption": 460
    },
    "2024-07-01": {
        "hardware": 300,
        "efficiency": 300,
        "pricing": 250,
        "adoption": 500,
        "note": "GPT-4o mini廉价版, LLaMA 3.1 405B最大开源"
    },
    "2024-07-16": {
        "hardware": 310,
        "efficiency": 310,
        "pricing": 260,
        "adoption": 530
    },
    "2024-08-01": {
        "hardware": 320,
        "efficiency": 320,
        "pricing": 270,
        "adoption": 560,
        "note": "Grok 2, 推理芯片(Groq/Cerebras)商用"
    },
    "2024-08-16": {
        "hardware": 330,
        "efficiency": 330,
        "pricing": 280,
        "adoption": 590
    },
    "2024-09-01": {
        "hardware": 350,
        "efficiency": 380,
        "pricing": 300,
        "adoption": 650,
        "note": "o1推理模型, 推理时计算范式改变, speculative decoding成熟"
    },
    "2024-09-16": {
        "hardware": 360,
        "efficiency": 400,
        "pricing": 310,
        "adoption": 680
    },
    "2024-10-01": {
        "hardware": 370,
        "efficiency": 420,
        "pricing": 320,
        "adoption": 700
    },
    "2024-10-16": {
        "hardware": 380,
        "efficiency": 430,
        "pricing": 330,
        "adoption": 720,
        "note": "Claude 3.5 Haiku, Agent框架兴起"
    },
    "2024-11-01": {
        "hardware": 390,
        "efficiency": 450,
        "pricing": 350,
        "adoption": 750
    },
    "2024-11-16": {
        "hardware": 400,
        "efficiency": 470,
        "pricing": 370,
        "adoption": 780,
        "note": "DeepSeek R1推理预览, 蒸馏技术成熟"
    },
    "2024-12-01": {
        "hardware": 400,
        "efficiency": 600,
        "pricing": 550,
        "adoption": 850,
        "note": "DeepSeek V3开源! 训练仅$5.6M, 价格降90%! o1正式版"
    },
    "2024-12-16": {
        "hardware": 430,
        "efficiency": 620,
        "pricing": 580,
        "adoption": 880
    },
    "2025-01-01": {
        "hardware": 450,
        "efficiency": 650,
        "pricing": 500,
        "adoption": 920,
        "note": "o3-mini推理突破, ARC-AGI, 蒸馏+量化成熟"
    },
    "2025-01-16": {
        "hardware": 470,
        "efficiency": 700,
        "pricing": 630,
        "adoption": 950,
        "note": "DeepSeek R1震撼全球, 推理成本再降80%"
    },
    "2025-02-01": {
        "hardware": 500,
        "efficiency": 720,
        "pricing": 650,
        "adoption": 980,
        "note": "Grok 3, GPT-4.5, HBM4量产前最后一波"
    },
    "2025-02-16": {
        "hardware": 520,
        "efficiency": 740,
        "pricing": 660,
        "adoption": 1000
    },
    "2025-03-01": {
        "hardware": 550,
        "efficiency": 780,
        "pricing": 700,
        "adoption": 1050,
        "note": "Claude 3.7 Sonnet+Gemini 2.5 Pro, 推理效率新高"
    },
    "2025-03-16": {
        "hardware": 570,
        "efficiency": 800,
        "pricing": 720,
        "adoption": 1080
    },
    "2025-04-01": {
        "hardware": 600,
        "efficiency": 850,
        "pricing": 750,
        "adoption": 1150,
        "note": "LLaMA 4开源+o3发布, Agent编程效率爆发"
    },
    "2025-04-16": {
        "hardware": 620,
        "efficiency": 880,
        "pricing": 770,
        "adoption": 1200,
        "note": "Gemini 2.5 Flash, FlashAttention-3成熟"
    },
    "2025-05-01": {
        "hardware": 680,
        "efficiency": 950,
        "pricing": 700,
        "adoption": 1300,
        "note": "Claude Opus 4+Sonnet 4深度编程, AI Coding工具全面成熟"
    },
    "2025-05-16": {
        "hardware": 700,
        "efficiency": 980,
        "pricing": 850,
        "adoption": 1350
    },
    "2025-06-01": {
        "hardware": 730,
        "efficiency": 1020,
        "pricing": 880,
        "adoption": 1400,
        "note": "o3-pro, 推理芯片开始分流GPU需求"
    },
    "2025-06-16": {
        "hardware": 750,
        "efficiency": 1050,
        "pricing": 900,
        "adoption": 1450
    },
    "2025-07-01": {
        "hardware": 780,
        "efficiency": 1080,
        "pricing": 920,
        "adoption": 1500
    },
    "2025-07-16": {
        "hardware": 800,
        "efficiency": 1100,
        "pricing": 940,
        "adoption": 1550
    },
    "2025-08-01": {
        "hardware": 800,
        "efficiency": 1200,
        "pricing": 800,
        "adoption": 1800,
        "note": "GPT-5发布! 百万token推理成本优化, OpenAI开源"
    },
    "2025-08-16": {
        "hardware": 920,
        "efficiency": 1220,
        "pricing": 1020,
        "adoption": 1850
    },
    "2025-09-01": {
        "hardware": 950,
        "efficiency": 1280,
        "pricing": 1050,
        "adoption": 1950,
        "note": "GPT-5 Codex, 推理芯片生态初步成熟"
    },
    "2025-09-16": {
        "hardware": 970,
        "efficiency": 1300,
        "pricing": 1080,
        "adoption": 2000
    },
    "2025-10-01": {
        "hardware": 1000,
        "efficiency": 1350,
        "pricing": 1100,
        "adoption": 2100,
        "note": "Agent框架成熟, Claude Haiku 4.5"
    },
    "2025-10-16": {
        "hardware": 1020,
        "efficiency": 1380,
        "pricing": 1120,
        "adoption": 2200
    },
    "2025-11-01": {
        "hardware": 1000,
        "efficiency": 1450,
        "pricing": 1000,
        "adoption": 2400,
        "note": "GPT-5.1+Gemini 3.0 Pro+Claude Opus 4.5, 三维共振"
    },
    "2025-11-16": {
        "hardware": 1100,
        "efficiency": 1480,
        "pricing": 1220,
        "adoption": 2450
    },
    "2025-12-01": {
        "hardware": 1150,
        "efficiency": 1550,
        "pricing": 1300,
        "adoption": 2600,
        "note": "GPT-5.2+Gemini 3.0 Flash, HBM4量产爬坡"
    },
    "2025-12-16": {
        "hardware": 1200,
        "efficiency": 1580,
        "pricing": 1350,
        "adoption": 2700
    },
    "2026-01-01": {
        "hardware": 1200,
        "efficiency": 1650,
        "pricing": 1400,
        "adoption": 2900,
        "note": "Gemini 2.0系列, HBM4量产加速"
    },
    "2026-01-16": {
        "hardware": 1350,
        "efficiency": 1680,
        "pricing": 1420,
        "adoption": 3000
    },
    "2026-02-01": {
        "hardware": 1500,
        "efficiency": 1780,
        "pricing": 1500,
        "adoption": 3300,
        "note": "Claude Opus 4.6+Grok 4+Gemini 3.1 Pro, 推理芯片全面商用"
    },
    "2026-02-16": {
        "hardware": 1600,
        "efficiency": 1850,
        "pricing": 1550,
        "adoption": 3500,
        "note": "GPT-5.3-Codex编程专用模型"
    },
    "2026-03-01": {
        "hardware": 2000,
        "efficiency": 2100,
        "pricing": 1200,
        "adoption": 4000,
        "note": "GPT-5.4+Capybara! 推理+编程+Agent三线突破"
    },
    "2026-03-16": {
        "hardware": 2100,
        "efficiency": 2150,
        "pricing": 1750,
        "adoption": 4200
    },
    "2026-04-01": {
        "hardware": 2500,
        "efficiency": 2500,
        "pricing": 2000,
        "adoption": 5000,
        "note": "GPT-5.5+DeepSeek V4! SWE-Bench 83.7%, 价格仅1/20! OpenClaw生态爆发"
    },
    "2026-04-16": {
        "hardware": 2600,
        "efficiency": 2550,
        "pricing": 2300,
        "adoption": 5200
    },
    "2026-05-01": {
        "hardware": 2800,
        "efficiency": 2700,
        "pricing": 2500,
        "adoption": 5800,
        "note": "Gemini Omni全模态+Claude Opus 4.8, AI Agent全面商用"
    },
    "2026-05-16": {
        "hardware": 2900,
        "efficiency": 2750,
        "pricing": 2550,
        "adoption": 6000
    },
    "2026-06-01": {
        "hardware": 3000,
        "efficiency": 2850,
        "pricing": 2600,
        "adoption": 6300,
        "note": "Claude Fable 5最新旗舰"
    },
    "2026-06-16": {
        "hardware": 3100,
        "efficiency": 2900,
        "pricing": 2650,
        "adoption": 6500,
        "note": "当前期(预估)"
    }
}


# ══════════════════════════════════════════════════════════════
# 2. PSO 权重优化器 (粒子群算法)
# ══════════════════════════════════════════════════════════════
# 寻找最优维度权重: 使得DVI与v3的AIC相关性最大
# 替代人为拍脑袋分配权重

class PSOOptimizer:
    """粒子群优化器 — 自动寻找最优维度权重"""
    
    def __init__(self, n_particles=50, n_iterations=100):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = 0.5  # 惯性权重
        self.c1 = 0.8  # 个体认知
        self.c2 = 0.9  # 社会认知
    
    def optimize(self, dim_data, target_values):
        """
        优化维度权重, 使得加权和与目标值最接近
        dim_data: list of dicts, 每期各维度评分
        target_values: list, 每期目标值 (v3的AIC作为目标)
        """
        n_dims = len(dim_data[0])
        n_periods = len(dim_data)
        
        # 初始化粒子
        particles = []
        for _ in range(self.n_particles):
            # 随机权重, 非负, 归一化和=1
            w = [random.random() for _ in range(n_dims)]
            s = sum(w)
            w = [wi / s for wi in w]
            particles.append({
                "position": w[:],
                "velocity": [0.0] * n_dims,
                "best_pos": w[:],
                "best_score": float("inf"),
            })
        
        # 全局最优
        gbest_pos = particles[0]["position"][:]
        gbest_score = float("inf")
        
        for iteration in range(self.n_iterations):
            # 衰减惯性权重
            w_curr = self.w * (1 - iteration / self.n_iterations)
            
            for p in particles:
                # 计算评分
                score = self._evaluate(p["position"], dim_data, target_values)
                
                # 更新个体最优
                if score < p["best_score"]:
                    p["best_score"] = score
                    p["best_pos"] = p["position"][:]
                
                # 更新全局最优
                if score < gbest_score:
                    gbest_score = score
                    gbest_pos = p["position"][:]
                
                # 更新速度和位置
                for d in range(n_dims):
                    r1, r2 = random.random(), random.random()
                    p["velocity"][d] = (
                        w_curr * p["velocity"][d]
                        + self.c1 * r1 * (p["best_pos"][d] - p["position"][d])
                        + self.c2 * r2 * (gbest_pos[d] - p["position"][d])
                    )
                    p["position"][d] += p["velocity"][d]
                    p["position"][d] = max(0.0, p["position"][d])  # 非负约束
                
                # 归一化权重 (防止全零)
                s = sum(p["position"])
                if s > 0:
                    p["position"] = [v / s for v in p["position"]]
        
        return {
            "weights": dict(zip(DVI_DIMS, [round(w, 4) for w in gbest_pos])),
            "score": round(gbest_score, 4),
            "iterations": self.n_iterations,
        }
    
    def _evaluate(self, weights, dim_data, target_values):
        """MSE误差"""
        error = 0.0
        for i, dim_vals in enumerate(dim_data):
            dvi = sum(w * dim_vals[d_idx] for d_idx, w in enumerate(weights))
            error += (dvi - target_values[i]) ** 2
        return error / max(len(dim_data), 1)


# ══════════════════════════════════════════════════════════════
# 3. 卡尔曼平滑器 (Kalman Smoother)
# ══════════════════════════════════════════════════════════════
# 去除噪声, 提取真实趋势

class KalmanSmoother:
    """一维卡尔曼平滑 — 用于DVI趋势信号去噪"""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.Q = process_noise  # 过程噪声
        self.R = measurement_noise  # 测量噪声
    
    def smooth(self, values):
        """前向-后向卡尔曼平滑"""
        n = len(values)
        if n == 0:
            return []
        
        # 前向滤波
        x_prior = [0.0] * n
        p_prior = [0.0] * n
        x_post = [0.0] * n
        p_post = [0.0] * n
        
        x = values[0]
        p = 1.0
        
        for t in range(n):
            # 预测
            x = x  # 恒定速度假设
            p = p + self.Q
            
            # 更新
            K = p / (p + self.R)
            x = x + K * (values[t] - x)
            p = (1 - K) * p
            
            x_post[t] = x
            p_post[t] = p
        
        # 后向平滑
        smoothed = [0.0] * n
        smoothed[-1] = x_post[-1]
        
        for t in range(n-2, -1, -1):
            C = p_post[t] / (p_post[t] + self.Q)
            smoothed[t] = x_post[t] + C * (smoothed[t+1] - x_post[t])
        
        return smoothed


# ══════════════════════════════════════════════════════════════
# 4. HMM 状态检测 (隐马尔可夫模型)
# ══════════════════════════════════════════════════════════════

class HMMStateDetector:
    """
    简化HMM — 检测AI发展的隐藏状态
    状态: (0)平台期 (1)加速期 (2)爆发期 (3)减速期
    观测: DVI增速
    """
    
    STATES = ["plateau", "accelerating", "explosive", "decelerating"]
    
    def __init__(self):
        # 状态转移矩阵 (4x4)
        self.transition = [
            [0.6, 0.3, 0.05, 0.05],  # plateau → ...
            [0.1, 0.5, 0.3, 0.1],    # accelerating → ...
            [0.0, 0.2, 0.5, 0.3],    # explosive → ...
            [0.3, 0.1, 0.1, 0.5],    # decelerating → ...
        ]
    
    def detect(self, dvi_history):
        """基于DVI增速检测状态序列"""
        n = len(dvi_history)
        if n < 2:
            return ["plateau"] * n
        
        # 计算增速: DVI(t) - DVI(t-1)
        deltas = []
        for i in range(1, n):
            delta = dvi_history[i] - dvi_history[i-1]
            deltas.append(delta)
        
        # 观测概率 (基于delta大小)
        states = ["plateau"]
        for delta in deltas:
            if delta > 100:
                states.append("explosive")
            elif delta > 40:
                states.append("accelerating")
            elif delta > 10:
                states.append("plateau")
            else:
                states.append("decelerating")
        
        return states


# ══════════════════════════════════════════════════════════════
# 5. 格兰杰因果检验 (简化版)
# ══════════════════════════════════════════════════════════════

class GrangerCausality:
    """简化格兰杰因果: 判断维度A的变化是否领先于维度B"""
    
    @staticmethod
    def test(dim_a, dim_b, max_lag=3):
        """
        测试dim_a是否格兰杰导致dim_b
        返回: 因果强度 (0-1), lag期
        """
        n = len(dim_a)
        if n < max_lag + 2:
            return 0, 0
        
        best_corr = 0
        best_lag = 0
        
        for lag in range(1, max_lag + 1):
            # 计算dim_a(t-lag)与dim_b(t)的相关性
            a_lagged = dim_a[:-lag] if lag < len(dim_a) else dim_a
            b_now = dim_b[lag:] if lag < len(dim_b) else dim_b
            
            if len(a_lagged) != len(b_now) or len(a_lagged) < 3:
                continue
            
            # Pearson相关系数
            n_p = len(a_lagged)
            mean_a = sum(a_lagged) / n_p
            mean_b = sum(b_now) / n_p
            
            num = sum((a - mean_a) * (b - mean_b) for a, b in zip(a_lagged, b_now))
            den = math.sqrt(sum((a - mean_a)**2 for a in a_lagged) * sum((b - mean_b)**2 for b in b_now))
            
            if den > 0:
                corr = abs(num / den)
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
        
        return best_corr, best_lag


# ══════════════════════════════════════════════════════════════
# 6. 全量计算
# ══════════════════════════════════════════════════════════════

def build_dvi_timeseries(weights=None):
    """构建DVI时间序列 — 等权平均 (数学最简, 无偏)"""
    if weights is None:
        # 等权平均: 每个维度权重相同, 最无偏的数学选择
        equal = 1.0 / len(DVI_DIMS)
        weights = {d: equal for d in DVI_DIMS}
    
    periods = []
    sorted_keys = sorted(DVI_HISTORY.keys())
    
    for key in sorted_keys:
        raw = DVI_HISTORY[key]
        dims = {k: raw.get(k, 0) for k in DVI_DIMS}
        dvi = sum(weights[d] * dims[d] for d in DVI_DIMS)
        periods.append({
            "date": key,
            "dvi": round(dvi),
            "dims": dims,
            "note": raw.get("note", ""),
        })
    
    return periods


def compare_v3_v4(v3_periods, dvi_periods):
    """对比v3能力指数和v4发展速度指数"""
    dvi_map = {p["date"]: p for p in dvi_periods}
    
    results = []
    for p in v3_periods:
        date = p["date"]
        if date in dvi_map:
            results.append({
                "date": date,
                "aic_v3": p["aic"],
                "dvi_v4": dvi_map[date]["dvi"],
                "aidi": p["aidi"],
            })
    return results


def run_full_v4():
    """全量分析"""
    print("=" * 70)
    print("AIDI v4 — 发展速度引擎 (Development Velocity Engine)")
    print("=" * 70)
    print()
    
    # 1. 读取v3能力数据
    from aidicore_v3 import build_timeseries_v3 as build_v3
    v3_periods = build_v3()
    
    # 2. 构建DVI
    dvi_periods = build_dvi_timeseries()
    
    # 3. PSO权重优化
    print("[PSO] 优化维度权重...")
    dim_data = [
        [
            p["dims"].get("hardware", 0),
            p["dims"].get("efficiency", 0),
            p["dims"].get("pricing", 0),
            p["dims"].get("adoption", 0),
        ]
        for p in dvi_periods
    ]
    target_aics = [p["aic"] for p in v3_periods]
    # Ensure same length
    n = min(len(dim_data), len(target_aics))
    dim_data = dim_data[:n]
    target_aics = target_aics[:n]
    
    pso = PSOOptimizer(n_particles=30, n_iterations=50)
    opt = pso.optimize(dim_data, target_aics)
    
    print(f"  最优权重: {opt['weights']}")
    print(f"  拟合误差: {opt['score']}")
    print()
    
    # 用最优权重重建DVI
    dvi_periods = build_dvi_timeseries(opt["weights"])
    
    # 4. 卡尔曼平滑DVI
    dvi_values = [p["dvi"] for p in dvi_periods]
    smoother = KalmanSmoother(process_noise=0.05, measurement_noise=0.15)
    dvi_smoothed = smoother.smooth(dvi_values)
    for i, p in enumerate(dvi_periods):
        p["dvi_smoothed"] = round(dvi_smoothed[i], 1)
    
    # 5. HMM状态检测
    detector = HMMStateDetector()
    dvi_raw = [p["dvi"] for p in dvi_periods]
    states = detector.detect(dvi_raw)
    for i, p in enumerate(dvi_periods):
        p["state"] = states[i] if i < len(states) else "unknown"
    
    # 6. 格兰杰因果分析
    print("[Causality] 维度间驱动关系:")
    for d1 in DVI_DIMS:
        for d2 in DVI_DIMS:
            if d1 == d2: continue
            v1 = [p["dims"][d1] for p in dvi_periods]
            v2 = [p["dims"][d2] for p in dvi_periods]
            strength, lag = GrangerCausality.test(v1, v2)
            if strength > 0.7:
                print(f"  {d1} → {d2}: strength={strength:.2f} lag={lag}")
    print()
    
    # 7. 输出结果
    print(f"{'日期':<14} {'DVI(速度)':>10} {'平滑后':>8} {'状态':<12} {'硬件':>6} {'效率':>6} {'价格':>6} {'渗透':>6}")
    print("-" * 70)
    
    # 只显示有状态的期 (加速/爆发)
    highlight_dates = set()
    for p in dvi_periods:
        if p["state"] in ["explosive", "accelerating"]:
            highlight_dates.add(p["date"])
    for p in dvi_periods:
        date = p["date"]
        dvi = p["dvi"]
        smooth = p["dvi_smoothed"]
        state = p["state"]
        dims = p["dims"]
        
        marker = " ★" if date in highlight_dates else "  "
        print(f"{date:<14} {dvi:>10} {smooth:>8} {state:<12}{dims['hardware']:>6} {dims['efficiency']:>6} {dims['pricing']:>6} {dims['adoption']:>6}")
    
    print()
    print("状态说明: explosive=爆发期 accelerating=加速期 plateau=平台期 decelerating=减速期")
    print()
    
    # 8. 对比v3
    print("[v3 vs v4 对比]")
    comparison = compare_v3_v4(v3_periods, dvi_periods)
    print(f"{'日期':<14} {'AIC(v3能力)':>12} {'DVI(v4速度)':>12} {'AIDI':>6}")
    print("-" * 50)
    # Show key periods
    key_dates = {"2022-12-01", "2023-03-01", "2024-05-01", "2024-12-01",
                 "2025-08-01", "2026-03-01", "2026-04-01", "2026-06-16"}
    for c in comparison:
        if c["date"] in key_dates:
            print(f"{c['date']:<14} {c['aic_v3']:>12} {c['dvi_v4']:>12} {c['aidi']:>+6}")
    
    # 9. 保存报告
    report = {
        "v4_weights": opt["weights"],
        "v4_pso_score": opt["score"],
        "v4_state_sequence": {p["date"]: p["state"] for p in dvi_periods},
        "comparison": comparison,
    }
    
    report_path = BASE / "reports" / "aidi_v4_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")
    
    return dvi_periods, v3_periods, opt


if __name__ == "__main__":
    run_full_v4()
