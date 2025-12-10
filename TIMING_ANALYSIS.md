# 详细时间统计和 Thinking 分析

## 概述

现在系统会记录每个 rollout 的详细时间统计，包括：

1. **每个 turn 的时间统计**：
   - LLM 生成时间（精确到毫秒）
   - LLM token 数量（输入/输出）
   - 工具执行时间
   - Turn 总时间

2. **Query 级别的时间统计**：
   - Query 总处理时间
   
3. **Group 级别的时间统计**：
   - Group 总时间
   - 平均 query 时间
   
4. **Step 级别的时间统计**：
   - Rollout 收集时间
   - Advantage 计算时间
   - Training 时间
   - Step 总时间

5. **Eval 级别的时间统计**：
   - Eval 总时间
   - 平均 query 时间

## 使用方法

### 1. 启用详细日志记录

训练时添加 `--enable-detailed-logging` 参数：

```bash
# 本地训练
python train_grpo_linksearch.py --mode masked --enable-detailed-logging

# Docker 训练
./scripts/docker_train_linksearch.sh --enable-detailed-logging
```

这会将详细的 rollout 日志保存到 `outputs/rollout_logs/` 目录。

### 2. 查看实时日志中的 thinking

如果你想在训练过程中实时查看模型的原始输出（包括可能的 thinking），设置环境变量：

```bash
# 在 .env 文件中添加
SHOW_RAW_OUTPUT="true"
```

或者在代码中设置：

```python
policy_config = PolicyConfig(
    show_raw_output=True,  # 显示模型原始输出
    verbose=True,
)
```

### 3. 分析已保存的日志

使用分析脚本来查看哪些 turn 有异常长的输出：

```bash
# 基本分析
python scripts/analyze_rollout_timing.py

# 显示 >= 500 token 的 turn 的原始输出预览
python scripts/analyze_rollout_timing.py --min-output-tokens 500 --show-raw-output

# 指定日志目录
python scripts/analyze_rollout_timing.py --logs-dir outputs/rollout_logs
```

### 4. 优化性能：减少 MAX_TOKENS

如果发现模型输出了大量 thinking 内容或者 token 数过多，可以降低 `MAX_TOKENS`：

```bash
# 在 .env 文件中
MAX_TOKENS="512"  # 从 4096 降低到 512
```

对于工具调用任务，512 或 1024 通常就足够了。

## 性能诊断

### 常见问题

#### 1. 输出 token 数异常高（> 500）

**症状**：一个简单的工具调用却输出了 700+ tokens

**可能原因**：
- 模型在输出工具调用前进行了 "thinking"
- System prompt 可能隐式鼓励了推理
- 模型本身有思考倾向（如 Qwen3）

**解决方案**：
1. 使用分析脚本查看原始输出：
   ```bash
   python scripts/analyze_rollout_timing.py --show-raw-output --min-output-tokens 500
   ```

2. 如果确认有 thinking，考虑：
   - 降低 `MAX_TOKENS` 限制输出长度
   - 修改 system prompt 明确要求"直接输出工具调用，不要思考"
   - 使用更激进的 `temperature` 设置

#### 2. LLM 生成速度慢（< 30 tokens/s on H200）

**症状**：H200 GPU 生成速度只有 16 tokens/s

**可能原因**：
1. `MAX_TOKENS` 设置过高（如 4096）
2. 输入序列过长
3. 模型未正确使用 GPU
4. 批处理大小为 1

**解决方案**：
1. **首要**：降低 `MAX_TOKENS` 到 512-1024
2. 检查是否使用了量化（`load_in_4bit=true`）
3. 使用 Flash Attention（Unsloth 应该自动启用）
4. 查看 GPU 利用率：`nvidia-smi`

## 日志文件结构

详细日志保存在 `outputs/rollout_logs/` 下：

```
outputs/rollout_logs/
├── step_1/
│   ├── query_q001/
│   │   ├── rollout_0.json
│   │   ├── rollout_1.json
│   │   └── ...
│   └── query_q002/
│       └── ...
└── step_2/
    └── ...
```

每个 JSON 文件包含：

```json
{
  "query_id": "q001",
  "step": 1,
  "rollout_index": 0,
  "query_total_time_ms": 45234.5,
  "turn_timings": [
    {
      "turn_number": 1,
      "llm_generation_time_ms": 1234.5,
      "llm_input_tokens": 1500,
      "llm_output_tokens": 150,
      "llm_raw_output": "...",  // 原始模型输出
      "llm_raw_output_length": 850,
      "tool_execution_time_ms": 5.2,
      "turn_total_time_ms": 1240.0
    }
  ],
  "tool_calls": [...],
  "reward": 1.5,
  "rubric": {...}
}
```

## 控制 Thinking

### 方法 1：System Prompt 修改

在 `link_search_agent/prompts.py` 中的 system prompt 添加：

```python
"""
IMPORTANT: Output tool calls directly without explanation or thinking.
Do not include reasoning or planning before tool calls.
"""
```

### 方法 2：使用更低的 temperature

降低 temperature 可以让模型输出更确定、更简洁：

```python
policy_config = PolicyConfig(
    enable_dynamic_temperature=False,  # 禁用动态 temperature
    base_temperature=0.3,  # 使用固定的低 temperature
)
```

### 方法 3：Post-processing 过滤

如果模型坚持输出 thinking，可以在解析时过滤掉：

```python
# 在 agent.py 的 _parse_tool_calls_from_response 中
# 只提取 <tool_call> 标签内的内容，忽略其他文本
```

## 性能基准

在 H200 + Qwen3-30B 上的预期性能：

| 配置 | Tokens/s | 说明 |
|------|----------|------|
| MAX_TOKENS=4096 | 15-20 | 太慢，不推荐 |
| MAX_TOKENS=1024 | 40-60 | 合理 |
| MAX_TOKENS=512 | 60-100+ | 推荐用于工具调用 |

## 示例输出

### 训练时的实时日志

```
⏱️  LLM Generation: 1234.56ms | Tokens: 1597 in / 150 out

🔍 Raw Model Output (850 chars, 150 tokens):
────────────────────────────────────────────────────────────────────────────────
<tool_call>{"name": "search_profile", "arguments": {"sql": "SELECT ..."}}</tool_call>
────────────────────────────────────────────────────────────────────────────────

⏱️  Tools Execution: 4.22ms
⏱️  Turn Total Time: 1240.00ms
```

### 分析脚本输出

```bash
$ python scripts/analyze_rollout_timing.py --min-output-tokens 500

📊 Found 24 rollout logs
================================================================================

📈 Performance Summary
================================================================================
Average tokens/second: 18.45
Average output tokens per turn: 523.2
Max output tokens in any turn: 772

🔍 Turns with >= 500 output tokens:
================================================================================

📍 Step 1, Query q001, Rollout 0, Turn 2
   Output tokens: 772
   LLM time: 47456.87ms
   Speed: 16.27 tokens/s
   Raw output length: 4523 chars
   ⚠️  Likely contains thinking (ratio: 22.6x)
   
   Raw Output Preview:
   --------------------------------------------------------------------------
   Let me think about this query. The user wants to find investment 
   managers in Munich with PE experience. I should search for...
   --------------------------------------------------------------------------
```

## 下一步

1. 使用 `--enable-detailed-logging` 运行训练
2. 使用分析脚本检查是否有 thinking
3. 根据分析结果调整 `MAX_TOKENS`
4. 如果需要，修改 system prompt 或添加 thinking 过滤
