"""
AIDI v5 — 预测引擎
==========================
双层架构: 算法层 × 语境层

├─ 算法层: Holt-Winters + ARIMA + 指数拟合 + Logistic + Bootstrap
│  → 产生 base_probability (纯数据驱动, 可复现)
│
├─ 语境层: 结构化调整因子 (LLM分析输入, 经验知识)
│  → 产生 adjustment_factors (显式记录, 可审计)
│
└─ 融合层: fused = base × contextual_adjustment
   → final_probability (算法+语境协同)

用法:
    from engine.aidicore_v5 import AIDIPredictor
    
    # 纯算法预测 (复现)
    pred = AIDIPredictor()
    pred.run()                     # 算法+默认语境调校
    
    # 带自定义语境调整
    pred.run(context_overrides={...})
    
    # 只看算法层
    pred.run(algorithm_only=True)
"""

import json, math, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DIMS_FILE = BASE_DIR / "data/curated/dim_scores.json"

# ── 六维定义 (与v3一致) ─────────────────────────────────────
DIMS = ["intelligence", "multimodal", "agent", "programming", "knowledge", "ecosystem"]
DIM_NAMES_ZH = {
    "intelligence": "语言智力", "multimodal": "多模态感知",
    "agent": "智能体行动力", "programming": "编程自改进",
    "knowledge": "知识系统", "ecosystem": "生态基础设施",
}
SYNERGY = {
    ("intelligence", "agent"): 0.35,  ("intelligence", "multimodal"): 0.20,
    ("intelligence", "programming"): 0.30, ("intelligence", "knowledge"): 0.25,
    ("agent", "programming"): 0.25,  ("agent", "ecosystem"): 0.20,
    ("agent", "knowledge"): 0.20,  ("multimodal", "programming"): 0.15,
    ("multimodal", "agent"): 0.20,  ("knowledge", "ecosystem"): 0.15,
    ("programming", "ecosystem"): 0.15,
}

# ── 静态数据 ───────────────────────────────────────────────
CNUS_PARITY_HISTORY = np.array([0.52, 0.55, 0.60, 0.65, 0.68, 0.72, 0.74, 0.76, 0.78])
ACTUAL_PRICES = np.array([60.0, 20.0, 15.0, 6.0, 3.0, 0.80, 0.60, 0.35, 0.18])
PRICE_FLOOR = 0.01


# ═══════════════════════════════════════════════════════════
#  算法层 (Algorithm Layer) — 纯统计, 可复现
# ═══════════════════════════════════════════════════════════

def _calc_aic(dim_scores):
    base = sum(dim_scores.values()) / len(dim_scores)
    total = sum(
        s * (min(dim_scores.get(d1, 0), dim_scores.get(d2, 0)) / 1000.0)
        * (dim_scores.get(d1, 0) * dim_scores.get(d2, 0) / 1_000_000.0) * 500
        for (d1, d2), s in SYNERGY.items()
        if dim_scores.get(d1, 0) > 0 and dim_scores.get(d2, 0) > 0
    )
    return round(base + total)


def holt_winters(y, h=25, alpha=0.5, beta=0.1, gamma=0.05, period=6):
    m, n_hist = period, len(y)
    level, trend, season = np.zeros(n_hist + h), np.zeros(n_hist + h), np.zeros(max(n_hist + h, m))
    pred = np.zeros(n_hist + h)
    level[0], trend[0] = y[0], y[1] - y[0] if n_hist > 1 else (y[-1] - y[0]) / n_hist
    for i in range(m): season[i] = y[i % n_hist]
    for t in range(1, n_hist):
        ps = season[t - m] if t >= m else 0
        level[t] = alpha * (y[t] - ps) + (1 - alpha) * (level[t-1] + trend[t-1])
        trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
        if t >= m: season[t] = gamma * (y[t] - level[t]) + (1 - gamma) * season[t - m]
        pred[t] = level[t-1] + trend[t-1] + (season[t-m] if t >= m else 0)
    for t in range(n_hist, n_hist + h):
        level[t], trend[t] = level[t-1] + trend[t-1], trend[t-1]
        season[t] = season[t - m] if t >= m else season[t % m]
        pred[t] = level[t] + trend[t] + season[t]
    residuals = y[1:] - pred[1:n_hist]
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return pred[n_hist:].tolist(), rmse, float(level[n_hist-1]), float(trend[n_hist-1]), residuals.tolist()


def arima_predict(y, p=2, q=2, h=25):
    d = np.diff(y); N = len(d); max_lag = max(p, q)
    Xl, Yl = [], []
    for t in range(max_lag, N):
        row = [d[t-i] if t-i >= 0 else 0 for i in range(1, p+1)] + [0]*q
        Xl.append(row); Yl.append(d[t])
    X, Yt = np.array(Xl), np.array(Yl)
    beta_ar = np.linalg.lstsq(X[:, :p], Yt, rcond=None)[0]
    resid = Yt - X[:, :p] @ beta_ar
    Xf = np.zeros((len(Yt), p+q))
    Xf[:, :p] = X[:, :p]
    for t in range(max_lag, N):
        for i in range(1, q+1):
            ri = t - max_lag - i
            Xf[t - max_lag, p+i-1] = resid[ri] if ri >= 0 else 0
    beta = np.linalg.lstsq(Xf, Yt, rcond=None)[0]
    sigma = float(np.std(Yt - Xf @ beta))
    d_ext, r_ext = list(d), list(Yt - Xf @ beta)
    preds = []
    for _ in range(h):
        idx = len(d_ext)
        val = sum(beta[i-1] * (d_ext[idx-i] if idx-i >= 0 else 0) for i in range(1, p+1))
        val += sum(beta[p+i-1] * (r_ext[idx-i] if 0 <= idx-i < len(r_ext) else 0) for i in range(1, q+1))
        d_ext.append(val); r_ext.append(0)
        preds.append(float(y[-1] + sum(d_ext[len(d)-1:])))
    return preds, (1.96 * sigma * np.sqrt(np.arange(1, h+1))).tolist()


def fit_exp_recent(y, recent_n=24):
    y_r, t = y[-recent_n:], np.arange(recent_n, dtype=float)
    c = np.polyfit(t, np.log(np.maximum(y_r, 1)), 1)
    return float(c[0]), float(np.exp(c[1]))


def exp_predict(y, h, recent_n=24):
    k, A = fit_exp_recent(y, recent_n)
    return (A * np.exp(k * np.arange(recent_n, recent_n + h, dtype=float))).tolist()


def logistic_model(t, K, a, b, t0):
    return K / (1 + a * np.exp(-b * (t - t0)))


def fit_logistic(y, t=None):
    if t is None: t = np.arange(len(y), dtype=float)
    try:
        return curve_fit(logistic_model, t, y, p0=[max(y)*3, 10, 0.5, len(y)/2], maxfev=10000)[0]
    except Exception:
        return None


def bootstrap_ci(y, h=25, n_bootstrap=2000, alpha=0.05):
    preds = []
    for _ in range(n_bootstrap):
        pc, _, _, _, resid = holt_winters(y, h)
        preds.append(np.array(pc) + np.random.choice(resid, h) * np.random.randn(h) * 0.3)
    preds = np.array(preds)
    return np.percentile(preds, alpha/2*100, axis=0).tolist(), np.percentile(preds, (1-alpha/2)*100, axis=0).tolist()


def detect_change_points(y, min_size=4, penalty=10):
    n, changes, pos = len(y), [0], min_size
    while pos < n - min_size:
        lm, rm = np.mean(y[changes[-1]:pos]), np.mean(y[pos:pos+min_size])
        cb = np.sum((y[changes[-1]:pos]-lm)**2) + np.sum((y[pos:pos+min_size]-rm)**2)
        cm = np.mean(y[changes[-1]:pos+min_size])
        if cb + penalty < np.sum((y[changes[-1]:pos+min_size]-cm)**2):
            changes.append(pos); pos += min_size
        else: pos += 1
    return changes[1:]


# ═══════════════════════════════════════════════════════════
#  语境层 (Context Layer) — LLM调校因子管理
# ═══════════════════════════════════════════════════════════
#
#  语境调整 = 算法看不到但实际世界存在的因素
#  每条因素明确记录: 来源, 作用方向, 量化幅度
#  不隐藏任何调整 — 完全可审计
# ═══════════════════════════════════════════════════════════

# 默认语境调校因子 (由LLM在当前对话中产生, 写入引擎)
# 格式: {assertion_id: [(factor_name, direction, magnitude, rationale)]}
#   direction: +1 = 增加概率, -1 = 降低概率
#   magnitude: 0.0~0.3 (单条不超过30%)
DEFAULT_CONTEXT_ADJUSTMENTS = {
    1: [  # Agent规模化爆发
        ("已有先例ChatGPT", +1, 0.08,
         "2022-12 ChatGPT本身就是一个DAU>1亿的'对话Agent'。"
         "现在行动力1580远超当时智力100, 但DAU天花板相同。"
         "有先例 = 门槛可复制."),
        ("编排层成熟度", +1, 0.05,
         "当前Agent失败率33%(OSWorld), 但MCP/Agent2Agent/编排框架"
         "正在快速成熟. 2026H2框架层突破概率高."),
        ("企业采用加速", +1, 0.05,
         "斯坦福2026: 企业AI采用率88%, 且正在从'辅助工具'转向'自主流程'. "
         "B端Agent > DAU门槛低."),
    ],
    2: [  # 推理成本再降50%
        ("线性外推的局限性", -1, 0.08,
         "纯指数外推假设降价速度不变, 但接近硬件成本下限($0.01~0.02)后减速. "
         "$0.18→$0.08是-55%, 而$60→$0.18已经跌了99.7%——"
         "剩余空间缩小, 降价难度增大."),
        ("竞争格局稳定化", -1, 0.05,
         "价格战阶段性缓和: OpenAI/Anthropic不再盲目追低价, "
         "转向差异化(推理+Agent能力). DeepSeek已接近成本价."),
        ("效率提升的时间差", +1, 0.03,
         "Epoch AI: 效率提升37%/年, 折算~6个月降价~18%. "
         "降价不止是价格战, 还有硬件效率的自然提升."),
    ],
    3: [  # 编程自改进成熟
        ("SWE-bench天花板效应", -1, 0.08,
         "SWE-bench已接近100%, 继续提升的空间变窄. "
         "但'自我改进'需要的不只是修bug, 而是架构级优化——"
         "这比SWE-bench高一个维度, 难度大很多."),
        ("交互效应阈值逻辑", +1, 0.05,
         "intel×prog交互效应当前43.8%, 远超8%的临界值——"
         "实际上已经有自我改进的雏形(Claude Code/Devin). "
         "'商用成熟'的定义是关键: 是从10%→100%, 而不是从0→1."),
        ("Agent执行链成熟", +1, 0.05,
         "行动力指数(1580→2865)大幅增长意味着AI不仅能设计改进方案, "
         "还能执行它. 自改进的三要素(智力+编程+行动力)都在加速."),
    ],
    4: [  # CN-US差距缩至15%
        ("美国政府管制", -1, 0.06,
         "对华芯片出口管制持续收紧, H100→B200→Rubin的封锁链在延长. "
         "中国拿不到最新硬件, 在纯能力巅峰上会被拉开."),
        ("效率优势抵消硬件劣势", +1, 0.08,
         "DeepSeek用H800训练出V4级模型, 证明了算法效率可以部分抵消硬件封锁. "
         "中美parity已从52%→78%, 封闭只会加速国产替代."),
        ("开源生态转移", +1, 0.05,
         "中国开源模型(DeepSeek R1 #1, Qwen #2)已经在全球领先. "
         "开源=美国的护城河在变薄."),
    ],
    5: [  # 普及率拐点
        ("Logistic拟合过度乐观", -1, 0.15,
         "Logistic S-curve预测97.7%采用率, 但现实中任何技术"
         "在>80%后会遇到'最后20%的难题'(数字鸿沟/基础设施/教育). "
         "PC/互联网从未达到97.7%."),
        ("成本门槛持续下降", +1, 0.08,
         "$0.08/1M的推理成本意味着个人开发者可以无成本试错——"
         "这是SaaS时代$0/月的等价物. 零边际成本驱动长尾采用."),
        ("社会接受度滞后", -1, 0.05,
         "斯坦福: 73%专家vs23%公众认为AI对工作有正面影响. "
         "信任缺口延迟采用."),
    ],
    6: [  # 能源议题
        ("已有法律先例加速", +1, 0.10,
         "California AI能耗披露法案(2025), EU AI Act能耗条款(2026生效). "
         "先例一旦建立, 跟风立法是中国和欧盟的典型模式."),
        ("执行力度可能不足", -1, 0.05,
         "出台法规是一回事, 真正执行是另一回事. "
         "AI竞赛中各国不愿因能源法规给自己加锁. "
         "象征性立法 > 实质性立法."),
        ("绿电选址已经发生", +1, 0.05,
         "冰岛/中东/北欧已经出现AI数据中心选址偏向绿电的实证. "
         "市场已经在响应, 政策的滞后性意味着法规只是追认现实."),
    ],
}


# ═══════════════════════════════════════════════════════════
#  融合层 (Fusion Layer) — 算法 × 语境
# ═══════════════════════════════════════════════════════════

def fuse_probability(base_p, adjustments):
    """算法概率 × 语境调整因子 = 融合概率
    
    每条adjustment可以是 ± 方向, 幅度累积但不越过[0.05, 0.95]边界
    融合公式: p_fused = clip(base_p × Π(1 + dir × mag), 0.05, 0.95)
    """
    p = float(base_p)
    for _, direction, magnitude, rationale in adjustments:
        p *= (1 + direction * magnitude)
    return round(max(0.05, min(0.95, p)), 3)


# ═══════════════════════════════════════════════════════════
#  AIDIPredictor — 主预测类
# ═══════════════════════════════════════════════════════════

class AIDIPredictor:
    """AIDI预测引擎 (v5) — 双层架构: 算法层 × 语境层"""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else BASE_DIR
        self.dims_file = self.base_dir / "data/curated/dim_scores.json"
        self.context_adjustments = DEFAULT_CONTEXT_ADJUSTMENTS
        self._load_data()
        self._build_timeseries()

    def _load_data(self):
        self._raw = json.loads(self.dims_file.read_text(encoding="utf-8"))
        self.periods_raw = self._raw["periods"]

    def _build_timeseries(self):
        keys = sorted(self.periods_raw.keys())
        periods, prev, base_raw = [], None, None
        for key in keys:
            raw = self.periods_raw[key]
            dims = ({k: v for k, v in raw["scores"].items() if k in DIMS}
                    if isinstance(raw, dict) and "scores" in raw
                    else {k: v for k, v in raw.items() if k in DIMS})
            aic_raw = _calc_aic(dims)
            if base_raw is None: base_raw = aic_raw
            aic = round(aic_raw / base_raw * 1000)
            aidi = aic - prev if prev is not None else 100
            periods.append({"date": key, "aic": aic, "aidi": aidi, "dimensions": dims.copy()})
            prev = aic
        self.periods = periods
        self.aic = np.array([p["aic"] for p in periods], dtype=float)
        self.aidi = np.array([p["aidi"] for p in periods], dtype=float)
        self.dates = [p["date"] for p in periods]
        self.n = len(self.aic)
        self.dims = {dim: np.array([p["dimensions"].get(dim, 0) for p in periods], dtype=float)
                     for dim in DIMS}

    # ── 公共AIC预测接口 ─────────────────────────────────────

    def predict_aic(self, horizon=12):
        """AIC三算法加权预测 (纯算法层, 可复现)"""
        h = horizon
        hw = holt_winters(self.aic, h)[0]
        arima_v = arima_predict(self.aic, h=h)[0]
        exp_v = exp_predict(self.aic, h)
        hw_low, hw_high = bootstrap_ci(self.aic, h)
        aic_hw, aic_arima, aic_exp = hw[-1], arima_v[-1], exp_v[-1]
        w_hw, w_arima, w_exp = 0.4, 0.3, 0.3
        agg = w_hw * aic_hw + w_arima * aic_arima + w_exp * aic_exp
        ci_low = w_hw * hw_low[-1] + w_arima * (aic_arima * 0.95) + w_exp * aic_exp * 0.85
        ci_high = w_hw * hw_high[-1] + w_arima * (aic_arima * 1.05) + w_exp * aic_exp * 1.15
        return {"holtwinters": round(aic_hw), "arima": round(aic_arima),
                "exponential": round(aic_exp), "aggregate": round(agg),
                "ci_95": [round(ci_low), round(ci_high)]}

    def predict_dimension(self, dim, horizon=25):
        """单维度Holt-Winters预测"""
        if dim not in DIMS:
            raise ValueError(f"Unknown dim: {dim}")
        if len(self.dims[dim]) < 6:
            return {"dim": dim, "current": int(self.dims[dim][-1]), "predictions": []}
        pred = holt_winters(self.dims[dim], horizon)[0]
        return {"dim": dim, "dim_zh": DIM_NAMES_ZH.get(dim, dim),
                "current": int(self.dims[dim][-1]),
                "predictions": [round(v) for v in pred]}

    def predict_all_dims(self, horizon=25):
        return {dim: self.predict_dimension(dim, horizon) for dim in DIMS}

    def detect_regime_changes(self):
        cps = detect_change_points(np.log(self.aic))
        return [self.dates[i] for i in cps[:10]] if cps else []

    # ── 断言引擎 (双层) ─────────────────────────────────────

    def get_assertions(self, horizon_2026=12, horizon_2027=25, algorithm_only=False):
        """生成6条断言预测 (双层: 算法层 + 语境调整)
        
        Args:
            algorithm_only: True=只输出算法基概率, 不做语境调校
            
        返回:
            每条断言包含 base_probability(算法层) 和 fused_probability(融合层)
        """
        h12, h25 = horizon_2026, horizon_2027
        dp = {dim: holt_winters(self.dims[dim], h25)[0] for dim in DIMS}

        def build(id_val, title, statement, base_p, raw_algorithm_desc,
                  critical, predicted, verification, adj_overrides=None):
            """构建单条断言, 融合算法层与语境层"""
            adjustments = adj_overrides if adj_overrides is not None else \
                         self.context_adjustments.get(id_val, [])
            fused = base_p if algorithm_only else fuse_probability(base_p, adjustments)

            entry = {
                "id": id_val,
                "title": title,
                "statement": statement,
                "algorithm_layer": {
                    "base_probability": round(base_p, 3),
                    "algorithm": raw_algorithm_desc,
                },
                "context_layer": {
                    "adjustments": [
                        {"factor": factor, "direction": dir, "magnitude": mag, "rationale": rat}
                        for factor, dir, mag, rat in adjustments
                    ],
                    "adjustment_count": len(adjustments),
                },
                "fused_probability": fused,
                "critical_value": critical,
                "predicted_value": predicted,
                "verifiable_by": verification,
            }
            return entry

        results = []

        # 1. Agent爆发
        aj, ej = dp["agent"][-1], dp["ecosystem"][-1]
        base_p1 = round(min(0.95, max(0.1, (aj/3000)*0.4 + (ej/2000)*0.3 + 0.85*0.3)), 3)
        results.append(build(1, "Agent规模化爆发",
            "出现DAU>1亿的Agent产品", base_p1,
            "Holt-Winters agent={}+ecosystem={} | agent_score={:.3f}".format(
                round(aj), round(ej), float(aj/3000*0.4 + ej/2000*0.3 + 0.85*0.3)),
            3000, round(aj),
            ["DAU>1亿Agent产品", "OSWorld>80%", "Agent错误率<15%"]))

        # 2. 定价
        coeffs_p = np.polyfit(np.arange(len(ACTUAL_PRICES)), np.log(ACTUAL_PRICES), 1)
        ft = np.arange(len(ACTUAL_PRICES), len(ACTUAL_PRICES) + h25 + 6)
        pred_prices = np.exp(np.polyval(coeffs_p, ft)).clip(min=PRICE_FLOOR).tolist()
        base_p2 = 0.80
        results.append(build(2, "推理成本再降50%",
            "最便宜前沿模型推理成本<$0.08/1M tokens", base_p2,
            "实际价格指数外推 半衰期={:.1f}期 k={:.4f}".format(
                np.log(2)/abs(coeffs_p[0]), coeffs_p[0]),
            0.08, round(pred_prices[6], 4),
            ["成本<$0.08/1M", "≥3家厂商标价<$0.10", "token付费新模式"]))

        # 3. 编程自改进
        i_pred, p_pred = dp["intelligence"][-1], dp["programming"][-1]
        s = SYNERGY[("intelligence", "programming")]
        weak, prod = min(i_pred, p_pred)/1000.0, (i_pred*p_pred)/1_000_000.0
        syn = s * weak * prod * 500
        syn_pct = min(100, syn / ((i_pred+p_pred)/2 + syn) * 100)
        base_p3 = round(min(0.85, max(0.2, (syn_pct - 4) / 8)), 3)
        results.append(build(3, "编程自改进成熟",
            "首款商用自我改进AI产品出现", base_p3,
            "intel×prog Holt-Winters intel={} prog={} synergy={:.1f}%".format(
                round(i_pred), round(p_pred), syn_pct),
            8.0, round(syn_pct, 2),
            ["intel×prog交互效应>8%", "自我改进商用产品", "SWE-bench 100%>1月"]))

        # 4. CN-US parity
        gap = 1 - CNUS_PARITY_HISTORY
        coeffs = np.polyfit(np.arange(len(gap)), np.log(gap), 1)
        ft4 = np.arange(len(gap), len(gap) + 6)
        pred_parity = 1 - np.exp(np.polyval(coeffs, ft4))
        cnus_val = round(float(pred_parity[-1] * 100), 1)
        base_p4 = round(min(0.85, max(0.3, (pred_parity[-1] - 0.82) / 0.08)), 3)
        results.append(build(4, "中美差距缩至15%以内",
            f"CN-US综合parity从{CNUS_PARITY_HISTORY[-1]*100:.0f}%→{cnus_val}%", base_p4,
            f"历史parity指数收敛 k={coeffs[0]:.4f}/半年 predicted={cnus_val}%",
            85, cnus_val,
            ["CN-US parity≥85%", "纯能力差距≤15%", "中国模型国际基准第一"]))

        # 5. 普及率拐点
        adopt_y = self.dims["ecosystem"] / max(self.dims["ecosystem"])
        popt = fit_logistic(adopt_y)
        if popt is not None and popt[0] > 1.2:
            ft5 = np.arange(len(adopt_y), len(adopt_y) + h25)
            adopt_pred = logistic_model(ft5, *popt)
            ratio = 0.53 / float(logistic_model(len(adopt_y)-1, *popt))
            adopt_pct = min(100, float(adopt_pred[-1] * ratio * 100))
            base_p5 = round(min(0.85, max(0.2, (adopt_pct/100 - 0.55) / 0.15)), 3)
        else:
            adopt_pct, base_p5 = 62.0, 0.50
        results.append(build(5, "AI普及率到达拐点",
            "全球AI采用率跨越65%", base_p5,
            f"Logistic S-curve + ecosystem proxy predicted={adopt_pct:.1f}%",
            65, round(adopt_pct, 1),
            ["采用率>65%", "企业采用率>90%", "日常使用AI>50%"]))

        # 6. 能源议题
        hw_rate = (self.dims["ecosystem"][-1] / self.dims["ecosystem"][-12])**(1/12) - 1
        eff_rate = 0.026
        gap_rate = float(hw_rate - eff_rate)
        base_p6 = round(min(0.75, max(0.15, 0.35 + gap_rate * 15)), 3)
        results.append(build(6, "AI能耗成为政治议题",
            "≥3个国家出台AI能源法规或碳税", base_p6,
            "需求增速{:.1f}%/半月-效率{:.1f}%/半月=缺口{:.2f}".format(
                hw_rate*100, eff_rate*100, gap_rate),
            0.1, round(gap_rate, 4),
            ["≥3国出台AI能源政策", "企业因能源调整战略", "选址偏向绿电"]))

        return results

    # ── 全量预测报告 ────────────────────────────────────────

    def predict_all(self, horizon_2026=12, horizon_2027=25, algorithm_only=False):
        pred_2026 = self.predict_aic(horizon_2026)
        pred_2027 = self.predict_aic(horizon_2027)
        assertions = self.get_assertions(horizon_2026, horizon_2027, algorithm_only)

        # 语境层摘要
        adj_count = sum(len(self.context_adjustments.get(i, [])) for i in range(1, 7))
        total_adj = sum(sum(abs(d*m) for _, d, m, _ in self.context_adjustments.get(i, [])) for i in range(1, 7))

        return {
            "meta": {
                "generated": "2026-06-21",
                "engine": "AIDI v5 Predictor",
                "architecture": "双层: 算法层(统计) × 语境层(LLM调校)",
                "algorithm_only": algorithm_only,
                "algorithms": ["Holt-Winters", "ARIMA(2,1,2)", "Exponential fit",
                               "Logistic S-curve", "Bootstrap CI", "Change Point Detection"],
                "context_adjustments_applied": adj_count,
                "total_adjustment_magnitude": round(total_adj, 3),
                "reproducibility_note": "算法层完全可复现; 语境层调整因子可审计",
            },
            "aic_predictions": {
                "current": {"date": self.dates[-1], "aic": int(self.aic[-1])},
                "2026-12-16": pred_2026,
                "2027-07-01": pred_2027,
            },
            "dim_predictions": self.predict_all_dims(horizon_2027),
            "change_points": self.detect_regime_changes(),
            "assertions": assertions,
        }

    def run(self, horizon_2026=12, horizon_2027=25, save=True, algorithm_only=False,
            context_overrides=None):
        """完整运行 + 打印 + 保存
        
        Args:
            algorithm_only: True=跳过语境调整, 只看算法
            context_overrides: {id: [(factor, dir, mag, rationale), ...]} 自定义覆盖
        """
        if context_overrides:
            self.context_adjustments.update(context_overrides)

        result = self.predict_all(horizon_2026, horizon_2027, algorithm_only)

        # ── 打印 ──
        mode_label = "纯算法层" if algorithm_only else "算法层 × 语境层(双层融合)"
        print("=" * 75)
        print(f"AIDI v5 — 预测引擎 [{mode_label}]")
        print("=" * 75)
        print(f"基线: {self.dates[0]} AIC={int(self.aic[0])}")
        print(f"当前: {self.dates[-1]} AIC={int(self.aic[-1])}")

        # AIC
        p26, p27 = result["aic_predictions"]["2026-12-16"], result["aic_predictions"]["2027-07-01"]
        print(f"\n{'方法':<35} {'2026-12-16 AIC':>18} {'2027-07-01 AIC':>18}")
        print("-" * 73)
        print(f"{'Holt-Winters':<35} {p26['holtwinters']:>18,} {p27['holtwinters']:>18,}")
        print(f"{'ARIMA(2,1,2)':<35} {p26['arima']:>18,} {p27['arima']:>18,}")
        print(f"{'指数增长(24期)':<35} {p26['exponential']:>18,} {p27['exponential']:>18,}")
        print(f"{'聚合预测':<35} {p26['aggregate']:>18,} {p27['aggregate']:>18,}")
        print(f"{'95% CI':<35} {p26['ci_95'][0]:>8,}~{p26['ci_95'][1]:>9,}  {p27['ci_95'][0]:>8,}~{p27['ci_95'][1]:>9,}")

        # 六维
        print(f"\n{'维度':<14} {'当前':>6} {'2026-12-16':>10} {'2027-07-01':>10}")
        print("-" * 42)
        for dim in DIMS:
            dp = result["dim_predictions"][dim]
            d26 = dp["predictions"][horizon_2026-1] if len(dp["predictions"]) >= horizon_2026 else 0
            d27 = dp["predictions"][-1] if dp["predictions"] else 0
            print(f"{DIM_NAMES_ZH.get(dim,dim):<14} {dp['current']:>6,} {d26:>10,} {d27:>10,}")

        # 变化点
        cps = result["change_points"]
        print(f"\n结构性变化点 ({len(cps)}个): {', '.join(cps[:5])}{'...' if len(cps)>5 else ''}")

        # 断言 (双层输出)
        print(f"\n{'#':<3} {'断言':<28} {'算法基概率':>10} {'融合概率':>10} {'语境调整':>9}")
        print("-" * 85)
        for a in result["assertions"]:
            base = a["algorithm_layer"]["base_probability"]
            fused = a["fused_probability"]
            adj_count = a["context_layer"]["adjustment_count"]
            arrow = "→" if fused != base else "="
            print(f"{a['id']:<3} {a['statement']:<28} {base*100:>8.1f}%  {arrow}  {fused*100:>7.1f}%  {adj_count:>3}条调校")

        # 每条断言的调整明细
        if not algorithm_only:
            print(f"\n语境调整明细:")
            for a in result["assertions"]:
                if a["context_layer"]["adjustment_count"] > 0:
                    print(f"\n  [{a['id']}] {a['title']} (基={a['algorithm_layer']['base_probability']*100:.0f}% → 融合={a['fused_probability']*100:.0f}%):")
                    for adj in a["context_layer"]["adjustments"]:
                        icon = "▲" if adj["direction"] > 0 else "▼"
                        print(f"    {icon} {adj['factor']} ({adj['direction']*adj['magnitude']:+.0%})")
                        print(f"       {adj['rationale'][:80]}...")

        if save:
            out = self.base_dir / "reports/aidi_v5_predictions.json"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n报告保存: {out}")

        return result


# ═══════════════════════════════════════════════════════════
#  Standalone entry
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    pred = AIDIPredictor()

    # 默认: 双层融合 (算法 × 语境)
    print("\n" + "█" * 75)
    print("█  模式A: 算法层 × 语境层 (双层融合)")
    print("█" * 75)
    pred.run(algorithm_only=False)

    # 可选: 只看算法层
    print("\n\n" + "█" * 75)
    print("█  模式B: 纯算法层 (可复现基线)")
    print("█" * 75)
    pred.run(algorithm_only=True, save=False)
