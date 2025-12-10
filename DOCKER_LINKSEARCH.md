# Link Search Agent Docker 训练指南

## 🐳 在 Docker 中训练 Link Search Agent

### 前置准备

1. **构建 Docker 镜像**（如果还没构建）：
```bash
cd /home/zhlmmc/rl-trl
bash scripts/build.sh
```

2. **准备数据库文件**：
确保你有 SQLite 数据库文件，包含 LinkedIn 个人资料数据。

3. **设置环境变量**：
```bash
# HuggingFace Token（必需）
export HF_TOKEN="your_huggingface_token"

# 数据库路径（可选，如果不设置会使用容器内的默认路径）
export PROFILE_DB_PATH="/home/zhlmmc/rl-people-search/link_search_agent/data/profiles.db"

# Wandb API Key（可选）
export WANDB_API_KEY="your_wandb_api_key"
```

### 方式 1: 使用自动脚本（推荐）

#### 启动容器
```bash
# 设置环境变量
export HF_TOKEN="your_token"
export PROFILE_DB_PATH="/path/to/profiles.db"

# 运行容器（会自动挂载数据库）
bash scripts/run.sh
```

容器启动后，会显示可用命令：
```
=== Qwen3-32B GRPO Training with TRL + Unsloth ===

Available training tasks:

1. Math Reasoning (GSM8K):
  - Run training: python train_grpo.py
  - With custom config: python train_grpo.py --config configs/custom.yaml

2. Link Search Agent:
  - Quick test: python train_grpo_linksearch.py --mode simple
  - Full training: python train_grpo_linksearch.py --mode masked
  - With detailed logs: python train_grpo_linksearch.py --mode masked --enable-detailed-logging
  - Or use script: ./scripts/train_linksearch.sh --mode masked

Other commands:
  - Test setup: python test_linksearch_setup.py
  - Run evaluation: python eval_model.py --checkpoint outputs/checkpoint-xxx
```

#### 在容器内开始训练

**快速测试**：
```bash
# 测试环境是否正确设置
python test_linksearch_setup.py

# 快速测试训练（simple 模式）
export TRAIN_DATASET_SIZE="50"
export EVAL_DATASET_SIZE="10"
export MAX_STEPS="20"
export WANDB_MODE="disabled"

python train_grpo_linksearch.py --mode simple
```

**完整训练**：
```bash
# 使用 masked 模式（推荐）
export TRAIN_DATASET_SIZE="1000"
export EVAL_DATASET_SIZE="100"
export MAX_STEPS="200"
export TARGET_ACCURACY="0.80"

python train_grpo_linksearch.py --mode masked
```

**使用训练脚本**：
```bash
# 使用便捷脚本
./scripts/train_linksearch.sh --mode masked

# 带详细日志
./scripts/train_linksearch.sh --mode masked --enable-detailed-logging

# 恢复训练
./scripts/train_linksearch.sh --mode masked --resume

# 从最佳 checkpoint 恢复
./scripts/train_linksearch.sh --mode masked --resume-best
```

### 方式 2: 手动运行 Docker

如果你想更细粒度地控制 Docker 运行：

```bash
# 手动运行容器
docker run -it --rm \
    --gpus all \
    --name qwen3-grpo-linksearch \
    --shm-size=32g \
    -v /home/zhlmmc/rl-trl:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v /home/zhlmmc/rl-people-search/link_search_agent/data/profiles.db:/workspace/link_search_agent/data/profiles.db:ro \
    -e HF_TOKEN="$HF_TOKEN" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e PROFILE_DB_PATH="/workspace/link_search_agent/data/profiles.db" \
    -p 6006:6006 \
    qwen3-grpo:latest
```

### 数据库挂载说明

数据库文件会被挂载到容器内的固定路径：
- **宿主机**: `$PROFILE_DB_PATH`（你设置的路径）
- **容器内**: `/workspace/link_search_agent/data/profiles.db`

容器内的环境变量 `PROFILE_DB_PATH` 会自动设置为容器内路径。

### 配置环境变量

在容器内，你可以设置以下环境变量来配置训练：

```bash
# 模型配置
export MODEL_NAME="unsloth/Qwen3-32B"

# 数据集配置
export TRAIN_DATASET_SIZE="1000"
export EVAL_DATASET_SIZE="100"
export HF_DATASET_ID="gboxai/linksearch"

# 训练参数
export MAX_STEPS="200"
export LEARNING_RATE="1e-5"
export PER_DEVICE_TRAIN_BATCH_SIZE="2"
export NUM_GENERATIONS="4"

# Agent 配置
export MAX_TURNS="15"
export MAX_PROFILES="10"

# 训练策略
export TARGET_ACCURACY="0.80"
export OUTPUT_DIR="outputs/grpo_linksearch_masked"

# Wandb
export WANDB_PROJECT="link-search-grpo"
export WANDB_MODE="online"  # 或 "disabled"
```

### 完整训练示例

```bash
# 1. 启动容器（在宿主机上）
export HF_TOKEN="your_token"
export PROFILE_DB_PATH="/home/zhlmmc/rl-people-search/link_search_agent/data/profiles.db"
export WANDB_API_KEY="your_wandb_key"

bash scripts/run.sh

# 2. 在容器内配置环境
export TRAIN_DATASET_SIZE="1000"
export EVAL_DATASET_SIZE="100"
export MAX_STEPS="200"
export TARGET_ACCURACY="0.80"
export WANDB_PROJECT="link-search-grpo"
export WANDB_NAME="exp-masked-qwen32b"

# 3. 开始训练
python train_grpo_linksearch.py --mode masked --enable-detailed-logging

# 或使用脚本
./scripts/train_linksearch.sh --mode masked --enable-detailed-logging
```

### 监控训练进度

**在容器内**：
```bash
# 查看训练日志
tail -f logs/training.log

# 查看 GPU 使用
nvidia-smi

# 查看输出目录
ls -la outputs/grpo_linksearch_masked/
```

**在宿主机上**（另一个终端）：
```bash
# 进入运行中的容器
docker exec -it qwen3-grpo-train bash

# 查看容器日志
docker logs qwen3-grpo-train

# 查看 GPU 使用
watch -n 1 nvidia-smi
```

**使用 Wandb**：
访问 https://wandb.ai/your-username/link-search-grpo

### 保存和恢复

训练输出会自动保存到宿主机的 `outputs/` 目录：
```bash
/home/zhlmmc/rl-trl/outputs/grpo_linksearch_masked/
├── checkpoint-10/
├── checkpoint-20/
├── ...
├── final/
└── rollout_logs/  (如果启用)
```

恢复训练：
```bash
# 在容器内
python train_grpo_linksearch.py --mode masked --resume

# 或从最佳 checkpoint
python train_grpo_linksearch.py --mode masked --resume_best
```

### 不同训练模式对比

| 模式 | 速度 | 准确度 | 适用场景 |
|------|------|--------|----------|
| simple | 最快 | 低 | 快速测试环境 |
| rollout | 中等 | 中等 | 实验和调试 |
| masked | 最慢 | 最高 | 完整训练（推荐） |

### 资源需求

**GPU 显存**：
- Qwen3-32B + 4-bit + LoRA-16: ~20-24GB
- 批次大小 2 + 梯度累积 4: ~28-32GB
- 建议：RTX 4090 / A6000 / A100

**磁盘空间**：
- 模型缓存: ~30GB
- 训练输出: ~10-20GB（取决于 checkpoint 数量）
- 数据库: ~1-10GB（取决于数据量）

### 故障排查

**问题 1: 找不到数据库**
```bash
# 检查数据库是否正确挂载
ls -la /workspace/link_search_agent/data/profiles.db

# 检查环境变量
echo $PROFILE_DB_PATH
```

**问题 2: HuggingFace 认证失败**
```bash
# 检查 token
echo $HF_TOKEN

# 重新设置
export HF_TOKEN="your_token"
huggingface-cli login
```

**问题 3: 显存不足**
```bash
# 减小批次大小
export PER_DEVICE_TRAIN_BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="8"

# 或使用更小的模型
export MODEL_NAME="unsloth/Qwen3-7B"
```

**问题 4: 容器无法访问 GPU**
```bash
# 检查 GPU 是否可用
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 性能优化

**加速训练**：
1. 增加 `gradient_accumulation_steps`（如果显存允许）
2. 使用 `--enable-detailed-logging` 仅在需要时
3. 减小 `MAX_TURNS`（如果任务允许）
4. 使用 SSD 存储数据库和输出

**节省显存**：
1. 减小 `PER_DEVICE_TRAIN_BATCH_SIZE`
2. 减小 `MAX_TOKENS`
3. 减小 `NUM_GENERATIONS`
4. 使用 4-bit 量化

### 多 GPU 训练

如果有多个 GPU：
```bash
# 使用所有 GPU
docker run --gpus all ...

# 使用特定 GPU
docker run --gpus '"device=0,1"' ...

# 在容器内检查
nvidia-smi
```

### 清理和维护

**清理容器**：
```bash
# 停止容器
docker stop qwen3-grpo-train

# 删除容器
docker rm qwen3-grpo-train

# 清理旧镜像
docker image prune
```

**清理输出**：
```bash
# 删除旧 checkpoint（保留最佳和最新）
cd /home/zhlmmc/rl-trl/outputs/grpo_linksearch_masked
ls checkpoint-* | head -n -2 | xargs rm -rf
```

### 下一步

1. **测试环境**: `python test_linksearch_setup.py`
2. **快速测试**: 使用 simple 模式训练 20 步
3. **完整训练**: 使用 masked 模式训练到目标 accuracy
4. **评估模型**: 评估训练好的模型
5. **部署应用**: 将模型应用到实际任务

### 有用的命令

```bash
# 查看容器资源使用
docker stats qwen3-grpo-train

# 查看容器内进程
docker top qwen3-grpo-train

# 复制文件到容器
docker cp local_file.txt qwen3-grpo-train:/workspace/

# 从容器复制文件
docker cp qwen3-grpo-train:/workspace/outputs/model.pt ./

# 查看容器日志
docker logs -f qwen3-grpo-train
```

## 📚 相关文档

- **快速入门**: `QUICKSTART_LINKSEARCH.md`
- **详细文档**: `LINKSEARCH_README.md`
- **迁移说明**: `MIGRATION_SUMMARY.md`
- **主文档**: `README.md`
