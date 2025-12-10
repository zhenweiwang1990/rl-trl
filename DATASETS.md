# 可用数据集说明

本项目支持多个公开数据集用于 GRPO 训练。以下是推荐的数据集列表：

## 🧮 数学推理数据集

### GSM8K (推荐)
- **数据集名称**: `openai/gsm8k`
- **配置**: `main`
- **任务**: 小学数学应用题
- **样本数**: 8.5K 训练样本
- **难度**: 需要 2-8 步推理
- **适用**: 数学推理、算术计算

**示例**:
```yaml
dataset_name: "openai/gsm8k"
```

**数据格式**:
```json
{
  "question": "Natalia sold clips to 48 of her friends...",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips... #### 24"
}
```

## 📝 文本摘要数据集

### TLDR (Reddit 摘要)
- **数据集名称**: `trl-internal-testing/tldr-preference-trl-style`
- **任务**: Reddit 帖子摘要
- **样本数**: ~120K 训练样本
- **格式**: prompt, chosen, rejected
- **适用**: 文本摘要、偏好学习

**示例**:
```yaml
dataset_name: "trl-internal-testing/tldr-preference-trl-style"
```

### OpenAI Summarization Feedback
- **数据集名称**: `openai/summarize_from_feedback`
- **配置**: `comparisons`
- **任务**: 带人类反馈的摘要
- **适用**: RLHF、偏好优化

**示例**:
```yaml
dataset_name: "openai/summarize_from_feedback"
```

## 🎯 使用方法

### 1. 修改配置文件

编辑 `configs/default.yaml`:

```yaml
# 使用 GSM8K 数学数据集
dataset_name: "openai/gsm8k"

# 或使用 TLDR 摘要数据集
# dataset_name: "trl-internal-testing/tldr-preference-trl-style"
```

### 2. 自定义奖励函数

根据不同数据集，需要调整 `train_grpo.py` 中的 `reward_function`：

#### GSM8K 奖励函数 (默认)
```python
def reward_function(samples, prompts, outputs, **kwargs):
    rewards = []
    for prompt, output in zip(prompts, outputs):
        # 检查是否包含答案标记 ####
        if "####" in output:
            reward = 1.0
        elif any(char.isdigit() for char in output):
            reward = 0.5  # 部分分数
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards
```

#### TLDR 摘要奖励函数
```python
def reward_function(samples, prompts, outputs, **kwargs):
    rewards = []
    for prompt, output in zip(prompts, outputs):
        # 基于摘要长度和质量评分
        length = len(output.split())
        if 10 <= length <= 50:  # 理想长度
            reward = 1.0
        elif length < 10:  # 太短
            reward = 0.3
        else:  # 太长
            reward = 0.6
        rewards.append(reward)
    return rewards
```

## 🔧 使用自定义数据集

### 数据格式要求

GRPO 训练需要以下格式之一：

#### 格式 1: Question-Answer (GSM8K 风格)
```json
{
  "question": "问题文本",
  "answer": "答案文本"
}
```

#### 格式 2: Prompt-Response
```json
{
  "prompt": "提示文本",
  "response": "响应文本"
}
```

#### 格式 3: Preference (RLHF 风格)
```json
{
  "prompt": "提示文本",
  "chosen": "更好的响应",
  "rejected": "较差的响应"
}
```

### 上传自定义数据集

1. 准备数据集（JSONL 或 CSV 格式）
2. 上传到 HuggingFace Hub：

```python
from datasets import Dataset, load_dataset

# 从本地文件加载
dataset = load_dataset("json", data_files="my_data.jsonl")

# 上传到 Hub
dataset.push_to_hub("your-username/your-dataset-name")
```

3. 在配置中使用：

```yaml
dataset_name: "your-username/your-dataset-name"
```

## 📊 数据集对比

| 数据集 | 任务类型 | 样本数 | 难度 | 推荐用途 |
|--------|---------|-------|------|---------|
| GSM8K | 数学推理 | 8.5K | 中等 | 推理能力训练 |
| TLDR | 文本摘要 | 120K | 简单 | 摘要生成 |
| Summarize Feedback | 摘要+RLHF | ~90K | 中等 | 偏好对齐 |

## 💡 提示

1. **首次使用**：GSM8K 是最简单的起点，数据集小且任务明确
2. **大规模训练**：TLDR 提供更多样本，适合长时间训练
3. **RLHF 场景**：使用带有人类反馈的数据集（如 Summarize Feedback）
4. **自定义任务**：准备自己的数据集并实现对应的奖励函数

## 🔗 相关链接

- [GSM8K 数据集](https://huggingface.co/datasets/openai/gsm8k)
- [TLDR 数据集](https://huggingface.co/datasets/trl-internal-testing/tldr-preference-trl-style)
- [Summarize Feedback](https://huggingface.co/datasets/openai/summarize_from_feedback)
- [TRL 文档](https://huggingface.co/docs/trl)

