# Link Search Agent Training with GRPO

这个模块支持训练 Link Search Agent 来搜索 LinkedIn 个人资料。Agent 使用 SQL 查询和工具调用来找到最相关的候选人。

## 🌟 特性

- ✅ 基于 GRPO (Group Relative Policy Optimization) 训练
- ✅ 支持 Qwen3-32B 大模型
- ✅ 使用 Unsloth 进行高效训练
- ✅ 工具调用支持（search_profile, read_profile, return_results）
- ✅ 详细的 rollout logging 用于调试
- ✅ Process rewards 用于改进搜索策略
- ✅ Wandb 集成用于监控训练

## 📋 数据集和数据库

### HuggingFace 数据集

Link Search Agent 使用 HuggingFace 数据集：`gboxai/linksearch`

数据集包含：
- 自然语言搜索查询
- 正确的 LinkedIn handles（ground truth）
- 训练集和测试集分割

### SQLite 数据库

此外还需要一个 SQLite 数据库包含 LinkedIn 个人资料数据。详细的数据库设置说明请查看 [DATABASE_SETUP.md](DATABASE_SETUP.md)。

**快速设置**：
```bash
# 如果有 PostgreSQL 数据库
bash scripts/generate_database.sh

# 或者使用现有 SQLite 数据库
cp /path/to/profiles.db link_search_agent/data/profiles.db
```

## 🚀 快速开始

### 1. 准备数据库

Link Search Agent 需要一个 SQLite 数据库包含 LinkedIn 个人资料。

**选项 A: 从 PostgreSQL 生成**（推荐）
```bash
# 1. 配置 PostgreSQL 连接
cp env.linksearch.example env.linksearch
nano env.linksearch  # 设置 PG_HOST, PG_USER, PG_PASSWORD, PG_DATABASE

# 2. 生成数据库
bash scripts/generate_database.sh
```

**选项 B: 使用现有数据库**
```bash
cp /path/to/profiles.db link_search_agent/data/profiles.db
```

详细说明请查看 [DATABASE_SETUP.md](DATABASE_SETUP.md)

数据库必须包含以下表：
- `profiles`: 个人资料信息（id, name, linkedin_handle, summary, about, skills）
- `experiences`: 工作经历
- `educations`: 教育背景

### 2. 设置 HuggingFace Token

```bash
export HF_TOKEN="your_huggingface_token"
export HF_DATASET_ID="gboxai/linksearch"
```

### 3. 运行训练

使用默认配置：
```bash
python train_grpo_linksearch.py --mode masked
```

使用自定义配置：
```bash
# 设置环境变量
export MODEL_NAME="unsloth/Qwen3-32B"
export TRAIN_DATASET_SIZE="1000"
export EVAL_DATASET_SIZE="100"
export MAX_STEPS="200"
export LEARNING_RATE="1e-5"
export PER_DEVICE_TRAIN_BATCH_SIZE="2"
export NUM_GENERATIONS="4"
export MAX_TURNS="15"
export MAX_PROFILES="10"
export TARGET_ACCURACY="0.80"
export OUTPUT_DIR="outputs/grpo_linksearch_masked"

# 运行训练
python train_grpo_linksearch.py --mode masked
```

### 4. 训练模式

有三种训练模式：

**masked (推荐)**：
- 使用完整的 agent rollout
- Token-level masking 只训练 agent 的输出
- Process rewards 用于中间步骤
- 最准确但最慢

```bash
python train_grpo_linksearch.py --mode masked
```

**rollout**：
- 使用完整的 agent rollout
- 使用 TRL 的标准 GRPO trainer
- 比 masked 快但准确度略低

```bash
python train_grpo_linksearch.py --mode rollout
```

**simple**：
- 使用启发式 reward function
- 最快但最不准确
- 用于快速测试

```bash
python train_grpo_linksearch.py --mode simple
```

## 📊 监控训练

### Wandb

训练会自动记录到 Wandb：
- 训练 loss 和 reward
- 评估 accuracy 和 score
- 搜索策略指标
- Rollout 时间

```bash
export WANDB_PROJECT="link-search-grpo"
export WANDB_NAME="experiment-1"
```

禁用 Wandb：
```bash
export WANDB_MODE="disabled"
```

### 详细日志

启用详细的 rollout logging：
```bash
python train_grpo_linksearch.py --mode masked --enable-detailed-logging
```

日志会保存到 `outputs/rollout_logs/` 目录，包含：
- 完整的对话历史
- 工具调用和结果
- 搜索到的 handles 和正确性
- Rubric 和 reward 计算

## 🎯 Reward Function

Link Search 使用复杂的 reward function 来鼓励好的搜索策略：

**基础分数**（0-1.5）：
- 基于找到的正确 handles 数量
- 完美匹配得 1.5 分

**策略奖励**：
- 早期发现正确结果：+0.15
- 零结果后拓宽搜索：+0.15
- 多结果后缩小搜索：+0.15
- 搜索后读取资料：+0.20

**惩罚**：
- 重复搜索：-0.10 每次
- 重复读取：-0.15 每次
- SQL 错误：-0.08 每次
- 无效读取：-0.10 每次

**严重错误**：
- 无法解析工具调用：-2.0
- 错误的工具名称：-1.8
- 错误的工具参数：-1.5

**完美执行奖励**：+3.0
- Score = 1.0
- 没有重复操作
- 没有错误
- 8 turns 内完成

## 📈 评估

训练过程中会定期评估：
- **Accuracy**: 百分比 score > 0.5
- **Average Score**: 找到正确 handles 的平均分数
- **Average Hits**: 平均找到的正确 handles 数量

当 accuracy 达到 target_accuracy（默认 80%）时，训练会自动停止。

## 🔧 恢复训练

从最新 checkpoint 恢复：
```bash
python train_grpo_linksearch.py --mode masked --resume
```

从特定 checkpoint 恢复：
```bash
python train_grpo_linksearch.py --mode masked --resume_from_checkpoint outputs/checkpoint-100
```

从最佳 checkpoint 恢复：
```bash
python train_grpo_linksearch.py --mode masked --resume_best
```

## 🐛 调试

### 查看训练日志

详细日志会打印每个 rollout 的：
- 查询和 gold handles
- 每一轮的工具调用
- 搜索结果和正确性
- 最终分数和 reward

### 检查 checkpoint

每个 checkpoint 包含：
- `adapter_model.safetensors`: LoRA 权重
- `training_state.json`: 训练状态
- `config.json`: 模型配置

最佳模型会被保存到 `best_model_path`。

## 💾 输出结构

```
outputs/grpo_linksearch_masked/
├── checkpoint-10/
│   ├── adapter_model.safetensors
│   ├── training_state.json
│   └── config.json
├── checkpoint-20/
├── final/
└── rollout_logs/  (如果启用详细日志)
    ├── step_10/
    │   ├── query_0/
    │   │   ├── rollout_0.json
    │   │   ├── rollout_1.json
    │   │   └── ...
    │   └── query_1/
    └── step_20/
```

## 🎓 训练提示

1. **开始小规模测试**：使用 simple 模式快速测试
2. **调整 max_turns**：如果 agent 经常超时，增加 max_turns
3. **调整 max_profiles**：根据查询难度调整目标数量
4. **监控 reward**：如果 reward 长期为负，检查数据质量
5. **使用详细日志**：启用 detailed logging 来理解 agent 行为

## 📝 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_NAME | unsloth/Qwen3-30B-A3B-128K | 基础模型 |
| TRAIN_DATASET_SIZE | 1000 | 训练数据集大小 |
| EVAL_DATASET_SIZE | 100 | 评估数据集大小 |
| MAX_STEPS | 200 | 最大训练步数 |
| LEARNING_RATE | 1e-5 | 学习率 |
| PER_DEVICE_TRAIN_BATCH_SIZE | 2 | 批次大小 |
| NUM_GENERATIONS | 4 | 每个 query 的 rollout 数 |
| BETA | 0.01 | KL 散度权重 |
| MAX_TURNS | 15 | Agent 最大轮数 |
| MAX_TOKENS | 4096 | 最大 tokens |
| MAX_PROFILES | 10 | 目标 profiles 数量 |
| TARGET_ACCURACY | 0.80 | 目标准确度 |
| OUTPUT_DIR | outputs/grpo_linksearch | 输出目录 |
| PROFILE_DB_PATH | link_search_agent/data/profiles.db | 数据库路径 |
| HF_TOKEN | - | HuggingFace token |
| HF_DATASET_ID | gboxai/linksearch | 数据集 ID |
| WANDB_PROJECT | link-search-grpo | Wandb 项目 |
| WANDB_MODE | online | Wandb 模式 |

## 📄 许可证

本项目使用 MIT 许可证。
