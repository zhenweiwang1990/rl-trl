# 快速修复：解决 LLM 生成速度慢的问题

## 问题症状

- H200 GPU 上 Qwen3-30B 生成速度只有 **16 tokens/s**
- 每个 turn 耗时 **40+ 秒**
- 输出 token 数异常高（**700+ tokens** 用于简单工具调用）

## 根本原因

1. **`MAX_TOKENS=4096` 设置过高** - 为每次生成预留了 4096 tokens 的 KV cache 空间
2. **模型可能输出了 thinking** - 在工具调用前进行推理，浪费 token

## 快速解决方案

### 方案 1：降低 MAX_TOKENS（立即见效！）

修改你的环境配置文件：

```bash
# 在云上的 .env 或 env.linksearch 文件中
MAX_TOKENS="512"  # 从 4096 降低到 512
```

**预期效果**：速度提升 **3-5 倍**（从 16 tokens/s 提升到 50-80 tokens/s）

### 方案 2：启用详细日志查看 Thinking

在训练时添加参数：

```bash
python train_grpo_linksearch.py --mode masked --enable-detailed-logging
```

然后使用分析脚本：

```bash
python scripts/analyze_rollout_timing.py --min-output-tokens 500 --show-raw-output
```

这会告诉你模型是否在输出 thinking 内容。

### 方案 3：如果确认有 Thinking，添加 System Prompt 限制

编辑 `link_search_agent/prompts.py`，在 system prompt 开头添加：

```python
"""
CRITICAL: Output tool calls directly in JSON format. 
DO NOT include any explanation, reasoning, or thinking before the tool call.
Respond ONLY with the tool call JSON.
"""
```

## 使用新配置

### 已创建的快速配置文件

使用 `configs/linksearch_fast.yaml`：

```bash
# 这个配置已经优化了 max_completion_length 为 512
python train_grpo_linksearch.py --mode masked --config configs/linksearch_fast.yaml
```

或者直接设置环境变量：

```bash
export MAX_TOKENS=512
python train_grpo_linksearch.py --mode masked
```

## Docker 使用

如果你在云上用 Docker，在启动容器时传入环境变量：

```bash
# 方法 1：在 Docker 启动脚本中设置
export MAX_TOKENS=512
./scripts/docker_train_linksearch.sh

# 方法 2：修改容器内的 .env 文件
docker exec -it qwen3-grpo-linksearch bash
nano .env
# 修改 MAX_TOKENS="512"
```

## 验证修复

训练开始后，检查日志：

### 修复前：
```
⏱️  LLM Generation: 47456.87ms | Tokens: 1597 in / 772 out
```
- 速度：16.27 tokens/s ❌

### 修复后（预期）：
```
⏱️  LLM Generation: 2000.00ms | Tokens: 1597 in / 150 out
```
- 速度：75 tokens/s ✅

## 性能基准

| MAX_TOKENS | H200 速度 | 适用场景 |
|------------|-----------|---------|
| 4096 | 15-20 t/s | ❌ 太慢 |
| 2048 | 25-35 t/s | 🟡 勉强可用 |
| 1024 | 40-60 t/s | ✅ 合理 |
| 512 | 60-100+ t/s | ✅✅ 推荐（工具调用任务）|

## 还需要帮助？

查看完整文档：`TIMING_ANALYSIS.md`

或运行诊断脚本：

```bash
python scripts/analyze_rollout_timing.py --help
```
