# AIDI Search Skill — AI发展指数搜索技能

## 用途
按日期范围搜索AI发展的六个维度的信息。
由GitHub Actions定时触发，搜索原始数据存到 `data/raw/`。

## 搜索维度

每个周期搜索6个维度：

### 1. 模型能力 (model)
- 新模型发布: 模型名称、参数量、架构
- 基准测试: MMLU/SWE-Bench/HumanEval/GPQA等分数
- 推理能力: 逻辑推理/数学/代码生成评测
- API发布: 新模型API上线时间、价格

### 2. 硬件算力 (hardware)
- GPU/TPU发布: NVIDIA/AMD/Google新品
- 算力价格: 云GPU租赁价格变化
- 产能: 台积电/三星先进制程产能
- 能效比: 每瓦算力提升

### 3. 算法创新 (algorithm)
- 架构突破: Transformer变体/新注意力机制/状态空间模型
- 训练方法: RLHF/DPO/自监督新进展
- 效率创新: 量化/蒸馏/MoE/稀疏化
- 多模态: 视觉/语音/视频统一模型

### 4. 商业生态 (business)
- 融资: AI公司融资轮次/估值
- 价格: API定价变化/降价/涨价
- 竞争格局: 市场份额/用户数
- 政策: AI监管/出口管制/法案

### 5. 应用渗透 (adoption)
- 用户数: ChatGPT/Claude/Gemini等MAU
- 企业采用: 500强企业AI采用率
- 开发者生态: API调用量/开发者数
- 垂直场景: 代码/医疗/法律/教育等

### 6. 开源生态 (opensource)
- 模型开源: Llama/Mistral/DeepSeek等
- GitHub热榜: 新星项目/Star增长
- 框架: PyTorch/TensorFlow/JAX更新
- 社区: HuggingFace模型数/下载量

## 搜索方式

```python
# 给定日期范围: start_date, end_date
# 对每个维度执行限定时间范围的搜索

results = {}
for dim in ["model", "hardware", "algorithm", "business", "adoption", "opensource"]:
    results[dim] = search(
        query=dim_query,
        date_from=start_date,
        date_to=end_date,
        max_results=20
    )
```

## 输出格式

```json
{
    "period": {"start": "2026-01-01", "end": "2026-01-15"},
    "searched_at": "2026-06-21T12:00:00Z",
    "dimensions": {
        "model": {
            "entries": [
                {"title": "...", "summary": "...", "source": "..."}
            ],
            "score": 750,
            "score_rationale": "DeepSeek V4预告使模型维度评分上升"
        },
        ...
    },
    "summary": "本期关键事件..."
}
```
