# Link Search Agent 快速入门

## 🐳 推荐：使用 Docker 训练

详细的 Docker 训练指南请查看：[DOCKER_LINKSEARCH.md](DOCKER_LINKSEARCH.md)

**快速开始（Docker）**：
```bash
# 1. 设置环境变量
export HF_TOKEN="your_token"
export PROFILE_DB_PATH="/home/zhlmmc/rl-people-search/link_search_agent/data/profiles.db"

# 2. 启动容器
bash scripts/run.sh

# 3. 在容器内测试
python test_linksearch_setup.py

# 4. 开始训练
./scripts/train_linksearch.sh --mode masked
```

---

## 🚀 本地运行（不推荐生产环境）

### 1. 环境设置

```bash
cd /home/zhlmmc/rl-trl

# 设置 HuggingFace token（用于下载数据集）
export HF_TOKEN="your_huggingface_token"

# 设置数据库路径（如果有本地数据库）
export PROFILE_DB_PATH="/path/to/profiles.db"

# 或者使用 rl-people-search 的数据库
export PROFILE_DB_PATH="/home/zhlmmc/rl-people-search/link_search_agent/data/profiles.db"
```

### 2. 安装依赖（如果还没安装）

```bash
pip install -r requirements.txt
```

### 3. 测试设置

```bash
python test_linksearch_setup.py
```

应该看到：
```
✓ link_search_agent imports successful
✓ grpo imports successful
✓ GRPO Config: model=unsloth/Qwen3-30B-A3B-128K, rollouts=4
✓ Policy Config: max_turns=15, max_profiles=10
✓ Found 3 tools: search_profile, read_profile, return_results
✓ All expected tools present
✓ Perfect case reward: 3.00
✓ Failure case reward: -0.20
✅ All tests passed!
```

### 4. 开始训练

#### 快速测试（Simple 模式）
```bash
export TRAIN_DATASET_SIZE="50"
export EVAL_DATASET_SIZE="10"
export MAX_STEPS="20"
export WANDB_MODE="disabled"

python train_grpo_linksearch.py --mode simple
```

#### 完整训练（Masked 模式 - 推荐）
```bash
export TRAIN_DATASET_SIZE="1000"
export EVAL_DATASET_SIZE="100"
export MAX_STEPS="200"
export TARGET_ACCURACY="0.80"
export WANDB_PROJECT="link-search-grpo"
export OUTPUT_DIR="outputs/grpo_linksearch_masked"

python train_grpo_linksearch.py --mode masked
```

#### 带详细日志的训练
```bash
python train_grpo_linksearch.py --mode masked --enable-detailed-logging
```

### 5. 监控训练

训练过程中会看到：
```
Step 1/200 | Loss: 0.5234 | Score: 0.450 | Reward: 0.820 | Rollout: 45.2s | Train: 2.1s

📊 Eval Step 10: Score=0.650, Accuracy=58.00%, Hits=6.50

🎯 New best accuracy: 58.00%
```

### 6. 断点恢复

如果训练中断，可以恢复：

```bash
# 从最新 checkpoint 恢复
python train_grpo_linksearch.py --mode masked --resume

# 从最佳 checkpoint 恢复
python train_grpo_linksearch.py --mode masked --resume_best

# 从指定 checkpoint 恢复
python train_grpo_linksearch.py --mode masked --resume_from_checkpoint outputs/checkpoint-100
```

## 📊 常用配置

### 小规模测试
```bash
export MODEL_NAME="unsloth/Qwen3-30B-A3B-128K"
export TRAIN_DATASET_SIZE="100"
export EVAL_DATASET_SIZE="20"
export MAX_STEPS="50"
export PER_DEVICE_TRAIN_BATCH_SIZE="2"
export NUM_GENERATIONS="4"
export MAX_TURNS="15"
export WANDB_MODE="disabled"

python train_grpo_linksearch.py --mode masked
```

### 中等规模训练
```bash
export MODEL_NAME="unsloth/Qwen3-32B"
export TRAIN_DATASET_SIZE="500"
export EVAL_DATASET_SIZE="50"
export MAX_STEPS="100"
export PER_DEVICE_TRAIN_BATCH_SIZE="2"
export NUM_GENERATIONS="4"
export TARGET_ACCURACY="0.80"

python train_grpo_linksearch.py --mode masked
```

### 完整训练
```bash
export MODEL_NAME="unsloth/Qwen3-32B"
export TRAIN_DATASET_SIZE="1000"
export EVAL_DATASET_SIZE="100"
export MAX_STEPS="200"
export PER_DEVICE_TRAIN_BATCH_SIZE="4"
export NUM_GENERATIONS="4"
export TARGET_ACCURACY="0.85"
export LEARNING_RATE="1e-5"

python train_grpo_linksearch.py --mode masked --enable-detailed-logging
```

## 🎯 训练目标

- **Accuracy > 80%**: 好的基础模型
- **Accuracy > 85%**: 优秀的模型
- **Accuracy > 90%**: 接近完美

训练会在达到 `TARGET_ACCURACY` 时自动停止。

## 📈 输出文件

训练完成后会生成：

```
outputs/grpo_linksearch_masked/
├── checkpoint-10/           # 每 10 步保存
├── checkpoint-20/
├── ...
├── final/                   # 最终模型
├── rollout_logs/           # 详细日志（如果启用）
│   ├── step_10/
│   │   ├── query_0/
│   │   │   ├── rollout_0.json
│   │   │   └── ...
│   │   └── query_1/
│   └── ...
```

## 🐛 故障排查

### 问题 1: 导入错误
```bash
python test_linksearch_setup.py
```

### 问题 2: 数据库找不到
```bash
export PROFILE_DB_PATH="/correct/path/to/profiles.db"
ls -la $PROFILE_DB_PATH  # 验证文件存在
```

### 问题 3: HuggingFace 认证失败
```bash
export HF_TOKEN="your_valid_token"
huggingface-cli login  # 或使用 CLI 登录
```

### 问题 4: CUDA 内存不足
```bash
# 减小批次大小
export PER_DEVICE_TRAIN_BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="8"

# 或使用更小的模型
export MODEL_NAME="unsloth/Qwen3-7B"
```

### 问题 5: Wandb 连接问题
```bash
# 禁用 Wandb
export WANDB_MODE="disabled"

# 或使用离线模式
export WANDB_MODE="offline"
```

## 📚 更多信息

- **详细文档**: `LINKSEARCH_README.md`
- **迁移说明**: `MIGRATION_SUMMARY.md`
- **主文档**: `README.md`
- **配置示例**: `configs/linksearch.yaml`

## 💡 提示

1. **先测试再训练**: 使用 simple 模式快速测试流程
2. **监控指标**: 关注 accuracy 和 avg_score
3. **调整参数**: 根据数据集调整 max_turns 和 max_profiles
4. **保存日志**: 使用 --enable-detailed-logging 来调试
5. **定期检查**: 查看 rollout_logs 了解 agent 行为

## ❓ 需要帮助？

如果遇到问题：
1. 查看 `LINKSEARCH_README.md` 的故障排查部分
2. 检查 `MIGRATION_SUMMARY.md` 确认文件都已正确迁移
3. 运行 `test_linksearch_setup.py` 验证环境
4. 查看详细日志了解 agent 行为

祝训练顺利！🎉
