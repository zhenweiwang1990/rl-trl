# Link Search Agent 迁移总结

## 概述

已成功将 `rl-people-search` 项目中的 Link Search Agent 代码迁移到 `rl-trl` 项目中，使其能够使用 TRL 训练 Qwen3-32B 模型来执行 LinkedIn 个人资料搜索任务。

## 迁移的文件

### 1. Link Search Agent 核心模块 (`link_search_agent/`)

- **`__init__.py`**: 模块导出
- **`config.py`**: 配置类（GRPOConfig, PolicyConfig）
- **`agent.py`**: LinkSearchAgent 主类，处理工具调用和对话
- **`tools.py`**: 工具函数（search_profile, read_profile）
- **`prompts.py`**: 系统提示词和工具 schema
- **`rollout.py`**: 评估 rubric 和 reward 计算
- **`grpo_utils.py`**: GRPO 训练辅助函数
- **`trainer.py`**: 自定义 LinkSearchGRPOTrainer
- **`rollout_logger.py`**: 详细的 rollout 日志记录
- **`data/__init__.py`**: 数据模块导出
- **`data/types.py`**: 数据类型定义
- **`data/query_loader.py`**: HuggingFace 数据集加载

### 2. GRPO 共享工具 (`grpo/`)

- **`__init__.py`**: 模块导出
- **`callbacks.py`**: AccuracyStopCallback 训练回调
- **`utils.py`**: 通用工具函数（checkpoint 管理等）

### 3. 训练脚本

- **`train_grpo_linksearch.py`**: Link Search Agent 训练脚本
  - 支持三种模式：masked（推荐）、rollout、simple
  - 支持断点恢复
  - 支持详细日志记录

### 4. 配置文件

- **`configs/linksearch.yaml`**: Link Search 专用配置

### 5. 文档

- **`LINKSEARCH_README.md`**: Link Search Agent 详细文档
- **`README.md`**: 更新主文档以包含 Link Search 任务
- **`MIGRATION_SUMMARY.md`**: 本文件

### 6. 依赖更新

- **`requirements.txt`**: 添加了必要的依赖
  - `pydantic>=2.0.0`: 配置管理
  - `modelscope>=1.32.0`: ModelScope 模型下载
  - `polars>=0.19.0`: 数据处理

## 项目结构

```
rl-trl/
├── link_search_agent/          # Link Search Agent 模块
│   ├── __init__.py
│   ├── agent.py               # Agent 实现
│   ├── config.py              # 配置类
│   ├── tools.py               # SQL 搜索工具
│   ├── prompts.py             # 提示词模板
│   ├── rollout.py             # Rollout 和 reward
│   ├── trainer.py             # 自定义 Trainer
│   ├── grpo_utils.py          # GRPO 辅助函数
│   ├── rollout_logger.py      # 日志记录
│   └── data/                  # 数据加载
│       ├── __init__.py
│       ├── types.py
│       └── query_loader.py
├── grpo/                       # GRPO 共享工具
│   ├── __init__.py
│   ├── callbacks.py           # 训练回调
│   └── utils.py               # 工具函数
├── train_grpo_linksearch.py   # Link Search 训练脚本
├── configs/
│   └── linksearch.yaml        # Link Search 配置
├── LINKSEARCH_README.md       # Link Search 文档
└── test_linksearch_setup.py   # 测试脚本
```

## 主要特性

### 1. Agent 能力
- SQL 搜索：使用 `search_profile` 在 SQLite 数据库中查询
- 资料读取：使用 `read_profile` 获取详细信息
- 结果返回：使用 `return_results` 提交最终答案
- 多轮对话：最多 15 轮交互

### 2. 训练模式
- **masked**: 完整 agent rollout + token-level masking（推荐）
- **rollout**: 完整 agent rollout + TRL 标准训练
- **simple**: 启发式 reward（快速测试）

### 3. Reward Function
- 基础分数：基于找到的正确 handles
- 策略奖励：鼓励好的搜索策略
- 过程奖励：为中间步骤提供反馈
- 惩罚：减少重复和错误

### 4. 详细日志
- 完整对话历史
- 工具调用和结果
- 搜索到的 handles 和正确性
- Rubric 和 reward 计算
- 保存为 JSON 格式便于分析

## 使用方法

### 快速开始

1. **设置环境变量**：
```bash
export PROFILE_DB_PATH="/path/to/profiles.db"
export HF_TOKEN="your_huggingface_token"
export MODEL_NAME="unsloth/Qwen3-32B"
```

2. **运行测试**：
```bash
python test_linksearch_setup.py
```

3. **开始训练**：
```bash
python train_grpo_linksearch.py --mode masked
```

### 训练选项

```bash
# 使用不同模式
python train_grpo_linksearch.py --mode masked
python train_grpo_linksearch.py --mode rollout
python train_grpo_linksearch.py --mode simple

# 启用详细日志
python train_grpo_linksearch.py --mode masked --enable-detailed-logging

# 断点恢复
python train_grpo_linksearch.py --mode masked --resume
python train_grpo_linksearch.py --mode masked --resume_from_checkpoint outputs/checkpoint-100
python train_grpo_linksearch.py --mode masked --resume_best
```

### 配置参数

可以通过环境变量配置：
- `TRAIN_DATASET_SIZE`: 训练集大小（默认 1000）
- `EVAL_DATASET_SIZE`: 评估集大小（默认 100）
- `MAX_STEPS`: 最大训练步数（默认 200）
- `LEARNING_RATE`: 学习率（默认 1e-5）
- `NUM_GENERATIONS`: 每个 query 的 rollout 数（默认 4）
- `MAX_TURNS`: Agent 最大轮数（默认 15）
- `MAX_PROFILES`: 目标 profiles 数量（默认 10）
- `TARGET_ACCURACY`: 目标准确度（默认 0.80）

## 依赖项

### 核心依赖
- `torch>=2.6.0`
- `transformers>=4.51.0`
- `trl>=0.12.0`
- `unsloth[cu124_ampere]>=2025.1`
- `peft>=0.14.0`
- `bitsandbytes>=0.44.0`

### Link Search 特定
- `pydantic>=2.0.0`: 配置管理
- `modelscope>=1.32.0`: 模型下载
- `polars>=0.19.0`: 数据处理
- `datasets>=3.0.0`: HuggingFace 数据集

### 训练工具
- `wandb>=0.19.0`: 实验跟踪
- `accelerate>=1.2.0`: 分布式训练

## 数据要求

### 1. SQLite 数据库
需要包含以下表：
- `profiles`: 个人资料（id, name, linkedin_handle, summary, about, skills）
- `experiences`: 工作经历
- `educations`: 教育背景

### 2. HuggingFace 数据集
- 数据集 ID: `gboxai/linksearch`
- 包含训练集和测试集
- 每条数据包含：查询文本和正确的 LinkedIn handles

## 训练流程

1. **数据加载**：从 HuggingFace 加载查询数据
2. **Rollout 收集**：每个查询生成多个 agent rollouts
3. **Advantage 计算**：使用 GRPO 计算 advantages
4. **Policy 更新**：使用 masked loss 更新模型
5. **评估**：定期评估模型性能
6. **保存 Checkpoint**：保存最佳模型

## 监控指标

### 训练指标
- `train/loss`: 训练 loss
- `train/policy_loss`: Policy loss
- `train/avg_reward`: 平均 reward
- `train/avg_score`: 平均分数
- `train/accuracy`: 训练准确度

### 评估指标
- `eval/accuracy`: 评估准确度（主要指标）
- `eval/avg_score`: 平均分数
- `eval/avg_hits`: 平均找到的正确 handles 数
- `eval/avg_reward`: 平均 reward

### Agent 指标
- 搜索策略质量
- 工具调用正确性
- 找到正确 handles 的轮次
- Token 使用量

## 输出文件

### Checkpoints
```
outputs/grpo_linksearch_masked/
├── checkpoint-10/
│   ├── adapter_model.safetensors  # LoRA 权重
│   ├── training_state.json        # 训练状态
│   └── config.json                # 模型配置
├── checkpoint-20/
└── final/                         # 最终模型
```

### Rollout 日志
```
outputs/grpo_linksearch_masked/rollout_logs/
├── step_10/
│   ├── query_0/
│   │   ├── rollout_0.json
│   │   ├── rollout_1.json
│   │   └── ...
│   └── query_1/
└── step_20/
```

## 故障排查

### 导入错误
运行测试脚本检查：
```bash
python test_linksearch_setup.py
```

### 数据库错误
确保 `PROFILE_DB_PATH` 正确设置：
```bash
export PROFILE_DB_PATH="/path/to/profiles.db"
```

### HuggingFace 错误
设置 token：
```bash
export HF_TOKEN="your_token"
```

### 显存不足
- 减小 `PER_DEVICE_TRAIN_BATCH_SIZE`
- 增加 `GRADIENT_ACCUMULATION_STEPS`
- 使用 4-bit 量化
- 减小 `MAX_TURNS` 或 `MAX_TOKENS`

## 下一步

1. **数据准备**：准备 SQLite 数据库和 HuggingFace 数据集
2. **环境配置**：设置所有必要的环境变量
3. **测试运行**：使用 simple 模式快速测试
4. **完整训练**：使用 masked 模式进行完整训练
5. **模型评估**：评估训练好的模型
6. **部署应用**：将模型应用到实际任务

## 注意事项

1. **数据质量**：确保数据库数据质量良好
2. **Reward 调优**：可能需要根据实际情况调整 reward function
3. **超参数**：根据数据集大小调整训练参数
4. **监控训练**：使用 Wandb 监控训练过程
5. **保存 checkpoint**：定期保存以防意外中断

## 参考资源

- **Link Search 文档**: `LINKSEARCH_README.md`
- **主文档**: `README.md`
- **配置示例**: `configs/linksearch.yaml`
- **测试脚本**: `test_linksearch_setup.py`
