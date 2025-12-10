# Qwen3-32B GRPO Training with TRL + Unsloth

这是一个使用 TRL (Transformer Reinforcement Learning) 和 Unsloth 训练 Qwen3-32B 模型的完整项目，支持 GRPO (Group Relative Policy Optimization) 训练方法。

## 🌟 特性

- ✅ 基于 NVIDIA PyTorch 25.11 容器
- ✅ 使用 Unsloth 进行高效训练（节省显存和加速）
- ✅ 支持 TRL 的 GRPO 训练方法
- ✅ 支持多种任务：
  - **数学推理**：使用 TRL 官方的 GSM8K GRPO 数据集
  - **Link Search Agent**：LinkedIn 个人资料搜索和匹配
- ✅ 支持 LoRA 微调
- ✅ 支持 4-bit 量化训练
- ✅ Wandb 集成用于实验跟踪
- ✅ 完整的训练、评估和交互测试脚本

## 📋 项目结构

```
rl-trl/
├── Dockerfile              # Docker 镜像配置
├── requirements.txt        # Python 依赖
├── train_grpo.py          # GRPO 训练脚本（数学推理）
├── train_grpo_linksearch.py  # Link Search Agent 训练脚本
├── eval_model.py          # 模型评估脚本
├── interactive_test.py    # 交互式测试脚本
├── configs/               # 配置文件
│   ├── default.yaml       # 默认配置（数学推理）
│   ├── linksearch.yaml    # Link Search 配置
│   └── custom.yaml        # 自定义配置示例
├── link_search_agent/     # Link Search Agent 模块
│   ├── agent.py          # Agent 实现
│   ├── config.py         # 配置
│   ├── tools.py          # 工具函数（SQL搜索）
│   ├── prompts.py        # 提示词模板
│   ├── rollout.py        # Rollout 和 reward 计算
│   ├── trainer.py        # 自定义 GRPO Trainer
│   └── data/             # 数据加载
├── grpo/                 # GRPO 通用工具
│   ├── callbacks.py      # 训练回调
│   └── utils.py          # 工具函数
├── scripts/               # 辅助脚本
│   ├── build.sh          # 构建 Docker 镜像
│   ├── run.sh            # 运行容器
│   └── train.sh          # 启动训练
├── outputs/              # 训练输出（checkpoints）
├── logs/                 # 训练日志
└── data/                 # 数据缓存
```

## 🚀 快速开始

### 方式 1: 使用 Docker（推荐）

#### 1. 构建 Docker 镜像

```bash
cd /home/zhlmmc/rl-trl
bash scripts/build.sh
```

#### 2. 运行容器

```bash
bash scripts/run.sh
```

#### 3. 在容器内开始训练

**数学推理任务（GSM8K）：**
```bash
# 使用默认配置
python train_grpo.py

# 使用自定义配置
python train_grpo.py --config configs/custom.yaml

# 使用命令行参数
python train_grpo.py --model unsloth/Qwen3-32B --load_in_4bit
```

**Link Search Agent 任务：**
```bash
# 使用 masked 模式（推荐）
python train_grpo_linksearch.py --mode masked

# 使用 rollout 模式
python train_grpo_linksearch.py --mode rollout

# 使用 simple 模式（快速测试）
python train_grpo_linksearch.py --mode simple

# 启用详细日志
python train_grpo_linksearch.py --mode masked --enable-detailed-logging
```

### 方式 2: 本地运行

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 开始训练

**数学推理：**
```bash
python train_grpo.py
```

**Link Search Agent：**
```bash
# 1. 准备数据库（如果需要）
bash scripts/generate_database.sh  # 从 PostgreSQL 生成
# 或
cp /path/to/profiles.db link_search_agent/data/profiles.db

# 2. 设置环境变量
export PROFILE_DB_PATH="/path/to/profiles.db"
export HF_TOKEN="your_huggingface_token"

# 3. 开始训练
python train_grpo_linksearch.py --mode masked
```

## 📊 训练配置

主要配置参数（在 `configs/default.yaml` 中）：

```yaml
# 模型设置
model_name: "unsloth/Qwen3-32B"
max_seq_length: 4096
load_in_4bit: true

# LoRA 设置
lora_r: 16
lora_alpha: 16

# 训练设置
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-5

# GRPO 设置
num_generations: 4
max_prompt_length: 1024
max_completion_length: 512
temperature: 0.7
beta: 0.01
```

## 🎯 使用不同数据集

默认使用 OpenAI 的 GSM8K 数学数据集（`openai/gsm8k`）。

### 可用数据集：

1. **GSM8K** (默认) - 数学推理
   ```yaml
   dataset_name: "openai/gsm8k"
   ```

2. **TLDR** - 文本摘要
   ```yaml
   dataset_name: "trl-internal-testing/tldr-preference-trl-style"
   ```

3. **Summarize Feedback** - RLHF 摘要
   ```yaml
   dataset_name: "openai/summarize_from_feedback"
   ```

详见 [DATASETS.md](DATASETS.md) 获取完整的数据集说明和使用指南。

要使用自己的数据集：

1. 修改 `configs/custom.yaml` 中的 `dataset_name`
2. 根据需要修改 `train_grpo.py` 中的 `reward_function`

## 📈 监控训练

### Wandb

项目默认使用 Wandb 进行训练监控：

```bash
# 登录 Wandb
wandb login

# 训练时会自动上传指标
python train_grpo.py
```

要禁用 Wandb：

```bash
python train_grpo.py --no_wandb
```

### TensorBoard

查看训练日志：

```bash
tensorboard --logdir outputs/qwen3-32b-grpo
```

## 🧪 评估和测试

### 评估模型

```bash
python eval_model.py --checkpoint outputs/qwen3-32b-grpo/final
```

### 交互式测试

```bash
python interactive_test.py --checkpoint outputs/qwen3-32b-grpo/final
```

## 🔧 高级用法

### 恢复训练

```bash
python train_grpo.py --resume
```

### 自定义训练参数

```bash
python train_grpo.py \
    --model unsloth/Qwen3-32B \
    --config configs/custom.yaml \
    --load_in_4bit
```

### 使用更长的上下文

在配置文件中修改：

```yaml
max_seq_length: 8192  # 或更大
```

## 💾 显存需求

| 配置 | 显存需求 | 推荐硬件 |
|------|---------|---------|
| 4-bit + LoRA-16 | ~20GB | RTX 4090, A6000 |
| 4-bit + LoRA-32 | ~24GB | RTX 4090, A6000 |
| FP16 + LoRA-16 | ~40GB | A100-40GB |
| FP16 + LoRA-32 | ~48GB | A100-80GB |

如果显存不足，可以：
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用 `load_in_4bit: true`
- 减小 `lora_r`

## 📝 关于 Qwen3-32B

Qwen3-32B 是 Qwen3 系列的旗舰模型，具有以下特点：

- 32.8B 参数
- 原生支持 32K 上下文，可扩展到 131K (使用 YaRN)
- 支持思维链模式切换
- 优秀的推理、代码和多语言能力

详见：https://huggingface.co/unsloth/Qwen3-32B

## 🐛 故障排查

### CUDA 内存不足

```bash
# 减小 batch size
python train_grpo.py --config configs/default.yaml
# 然后修改 configs/default.yaml:
# per_device_train_batch_size: 1
# gradient_accumulation_steps: 8
```

### Transformers 版本错误

```bash
pip install transformers>=4.51.0
```

### Unsloth 安装问题

```bash
pip install "unsloth[cu124_ampere]>=2025.1" --upgrade
```

## 📄 许可证

本项目使用 MIT 许可证。

## 📚 任务文档

- **数学推理（GSM8K）**：见本 README
- **Link Search Agent**：见 [LINKSEARCH_README.md](LINKSEARCH_README.md)

## 🙏 致谢

- [TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [Unsloth](https://github.com/unslothai/unsloth) - 高效训练加速
- [Qwen Team](https://github.com/QwenLM/Qwen) - Qwen3 模型

## 📮 联系方式

如有问题或建议，欢迎提交 Issue。
