# 快速开始指南

## 10 分钟开始训练 Qwen3-32B

### 1️⃣ 构建 Docker 镜像 (5 分钟)

```bash
cd /home/zhlmmc/rl-trl
bash scripts/build.sh
```

### 2️⃣ 配置环境变量 (可选)

```bash
# 复制环境变量模板
cp env.example .env

# 编辑 .env 文件，添加你的 API keys
vim .env
```

如果不使用 Wandb，可以跳过这一步。

### 3️⃣ 启动容器并开始训练 (自动)

```bash
# 启动容器（会自动进入交互模式）
bash scripts/run.sh

# 在容器内，使用默认配置开始训练
python train_grpo.py

# 或者使用自定义配置
python train_grpo.py --config configs/custom.yaml
```

### 4️⃣ 监控训练

训练开始后，你可以：

- 在终端查看实时日志
- 访问 Wandb 查看详细指标：https://wandb.ai
- 查看本地日志：`tail -f logs/training.log`

### 5️⃣ 评估模型

训练完成后：

```bash
# 评估最终模型
python eval_model.py --checkpoint outputs/qwen3-32b-grpo/final

# 交互式测试
python interactive_test.py --checkpoint outputs/qwen3-32b-grpo/final
```

## 🎯 常见使用场景

### 场景 1: 使用 4-bit 量化节省显存

```bash
python train_grpo.py --load_in_4bit
```

### 场景 2: 恢复训练

```bash
python train_grpo.py --resume
```

### 场景 3: 不使用 Wandb

```bash
python train_grpo.py --no_wandb
```

### 场景 4: 自定义模型

```bash
python train_grpo.py --model unsloth/Qwen3-32B --config configs/custom.yaml
```

## 💡 提示

1. **显存不足？** 
   - 编辑 `configs/default.yaml`
   - 减小 `per_device_train_batch_size: 1`
   - 增加 `gradient_accumulation_steps: 8`

2. **加速训练？**
   - 使用 4-bit 量化：`load_in_4bit: true`
   - 启用 gradient checkpointing（默认开启）

3. **更好的效果？**
   - 增加 LoRA rank：`lora_r: 32`
   - 更多训练轮次：`num_train_epochs: 5`
   - 更大的 batch size

## 📊 训练时间估算

| 配置 | GPU | 时间/Epoch |
|------|-----|-----------|
| 4-bit + LoRA-16 | RTX 4090 | ~2 小时 |
| 4-bit + LoRA-16 | A100-40GB | ~1 小时 |
| FP16 + LoRA-32 | A100-80GB | ~1.5 小时 |

## ❓ 遇到问题？

查看 [README.md](README.md) 的故障排查部分，或查看日志：

```bash
cat logs/training.log
```

## 🎉 完成！

训练完成后，你的模型将保存在 `outputs/qwen3-32b-grpo/final/`

享受你的 Qwen3-32B 模型吧！🚀
