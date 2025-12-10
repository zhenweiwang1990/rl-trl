# RL-TRL 训练流程改进计划

基于 rl-unsloth 项目的最佳实践，改进 rl-trl 的训练评估、保存、断点续训等功能。

## 📋 改进总览

### 1️⃣ 优先级 P0 - 核心功能增强

#### 1.1 Checkpoint 保存增强
**当前状态**:
- ✅ 已保存: model, tokenizer, training_state.json
- ❌ 缺失: optimizer 状态, training_metadata.json, evals_without_improvement

**改进内容**:
```python
# _save_checkpoint() 增强
1. 保存 optimizer 状态到 optimizer.pt
2. 创建独立的 training_metadata.json 文件（包含详细的 metrics）
3. 在 training_state.json 中添加 evals_without_improvement 字段
4. 保存更多 metrics 到 metadata（包括 avg_score, avg_hits 等）
```

**受益**:
- 完整恢复训练状态，包括优化器动量
- 更准确的断点续训
- 更详细的 checkpoint 元数据供后续分析

#### 1.2 Checkpoint 加载增强
**当前状态**:
- ✅ 已加载: global_step, best_accuracy, best_model_path
- ❌ 缺失: optimizer 状态, evals_without_improvement

**改进内容**:
```python
# _load_checkpoint() 增强
1. 加载 optimizer 状态（如果存在）
2. 加载 evals_without_improvement（用于 early stopping）
3. 打印更详细的恢复信息
```

#### 1.3 评估日志保存
**当前状态**:
- ❌ 评估结果只打印到控制台，未保存到文件
- ❌ 缺少详细的 rubric 统计

**改进内容**:
```python
# _run_evaluation() 增强
1. 创建 eval_logs/ 目录
2. 保存每次评估结果到 eval_step_XXXX.json
3. 添加详细统计:
   - 基本统计: step, accuracy, correct_answers, total_samples
   - 奖励统计: avg_reward, median_reward, std_reward, min/max_reward
   - 详细指标: 
     * attempted_answer (尝试回答的数量)
     * found_correct_profile (找到正确 profile 的数量)
     * read_correct_profile (读取正确 profile 的数量)
     * avg_turns (平均轮数)
     * avg_search_attempts (平均搜索次数)
   - 评估时间: eval_time
```

**JSON 格式示例**:
```json
{
  "step": 10,
  "accuracy": 0.75,
  "correct_answers": 75,
  "total_samples": 100,
  "attempted_answer": 95,
  "avg_reward": 0.623,
  "median_reward": 0.750,
  "std_reward": 0.412,
  "min_reward": -1.0,
  "max_reward": 1.5,
  "found_correct_profile": 80,
  "read_correct_profile": 78,
  "avg_turns": 4.2,
  "avg_search_attempts": 2.8,
  "eval_time": 234.5
}
```

#### 1.4 Baseline 评估增强
**当前状态**:
- ✅ 已有 run_baseline_eval 标志和基本实现
- ❌ Baseline 结果未保存
- ❌ 无法跳过 baseline（重复运行浪费时间）

**改进内容**:
```python
# train() 方法中的 baseline 评估增强
1. 保存 baseline 结果到 baseline_eval.json
2. 添加时间戳和完整统计
3. 如果 RUN_BASELINE_EVAL=false 且 baseline_eval.json 存在，则从文件加载
4. 打印更清晰的 baseline 结果
```

---

### 2️⃣ 优先级 P1 - 日志和监控增强

#### 2.1 训练步骤详细日志
**当前状态**:
- ✅ 基本的步骤日志
- ❌ 缺少阶段划分和详细统计

**改进内容**:
```python
# train() 主循环增强
详细模式（VERBOSE=true）下显示:
1. STEP 开始标记
2. Rollout 阶段日志
3. Advantage 计算阶段日志
4. Backpropagation 阶段日志
5. Group 统计总结:
   - Total groups: X
   - Groups kept for training: Y
   - Groups filtered (low variance): Z
   - Rollouts finished early: N/M
   - Total rollout time: Xs
   - Total training time: Ys
   - Trainable tokens: X/Y (Z%)
```

#### 2.2 评估结果增强打印
**当前状态**:
- ✅ 已有基本的评估日志打印
- ✅ 已实现详细的开始和结束日志

**改进内容**:
```python
# _run_evaluation() 打印增强
1. 添加中位数、标准差、最小/最大奖励
2. 添加详细 rubric 统计
3. 添加评估耗时
4. 保存位置提示
```

#### 2.3 Wandb 指标补充
**当前状态**:
- ✅ 基本的 train/eval 指标
- ❌ 缺少详细的 rubric 指标

**改进内容**:
```python
# Wandb 日志增强
训练阶段新增:
- train/median_reward
- train/groups_kept
- train/groups_filtered
- train/num_early_exit
- train/trainable_token_ratio

评估阶段新增:
- eval/median_reward
- eval/std_reward
- eval/attempted_answer_rate
- eval/found_profile_rate
- eval/read_profile_rate
- eval/avg_turns
- eval/avg_search_attempts
```

---

### 3️⃣ 优先级 P2 - 高级功能（可选）

#### 3.1 TrainingMetrics 字段补充
**改进内容**:
```python
# 在 TrainingMetrics 中添加（已在 utils.py 定义但未充分使用）:
- avg_score: float (当前的 rubric score)
- avg_hits: float (平均找到的正确 handle 数)
```

#### 3.2 Training Summary（训练结束总结）
**改进内容**:
```python
# train() 方法结束时
1. 打印训练总结:
   - 总训练步数
   - 最佳准确率
   - 最佳模型路径
   - 总训练时间
   - 是否达到目标准确率
2. 如果有 early stopping，说明原因
```

---

## 🔧 具体实现步骤

### Step 1: 增强 TrainingMetrics（rl-trl/link_search_agent/trainer.py）
```python
# 在 TrainingMetrics dataclass 中添加缺失字段
@dataclass
class TrainingMetrics:
    # ... 现有字段 ...
    avg_score: float = 0.0  # 已有
    avg_hits: float = 0.0   # 已有
    median_reward: float = 0.0  # 新增
    # 其他字段已经在 grpo/utils.py 中定义
```

### Step 2: 增强 _save_checkpoint
```python
def _save_checkpoint(self, metrics: TrainingMetrics):
    """Save model checkpoint and training state."""
    checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save model
    self.model.save_pretrained(str(checkpoint_dir))
    self.tokenizer.save_pretrained(str(checkpoint_dir))
    
    # 2. Save optimizer state (NEW)
    torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    
    # 3. Save training state (ENHANCED)
    training_state = {
        "global_step": self.global_step,
        "best_accuracy": self.best_accuracy,
        "best_model_path": str(self.best_model_path) if self.best_model_path else None,
        "evals_without_improvement": self.evals_without_improvement,  # NEW
    }
    with open(checkpoint_dir / "training_state.json", 'w') as f:
        json.dump(training_state, f, indent=2)
    
    # 4. Save training metadata (NEW)
    training_metadata = {
        "step": self.global_step,
        "accuracy": metrics.accuracy,
        "metrics": {
            "loss": metrics.loss,
            "policy_loss": metrics.policy_loss,
            "kl_loss": metrics.kl_loss,
            "avg_reward": metrics.avg_reward,
            "median_reward": metrics.median_reward,
            "avg_score": metrics.avg_score,
            "avg_hits": metrics.avg_hits,
            "reward_std": metrics.reward_std,
        }
    }
    with open(checkpoint_dir / "training_metadata.json", 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    logger.info(f"💾 Model and training state saved to: {checkpoint_dir}")
    return checkpoint_dir
```

### Step 3: 增强 _load_checkpoint
```python
def _load_checkpoint(self, checkpoint_path: Path):
    """Load training state from checkpoint."""
    logger.info(f"📂 Loading training state from checkpoint: {checkpoint_path}")
    
    # 1. Load optimizer state (NEW)
    optimizer_path = checkpoint_path / "optimizer.pt"
    if optimizer_path.exists():
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        logger.info("✓ Optimizer state loaded")
    else:
        logger.warning("⚠️  Optimizer state not found, starting with fresh optimizer")
    
    # 2. Load training state (ENHANCED)
    state_file = checkpoint_path / "training_state.json"
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        self.global_step = state.get("global_step", 0)
        self.best_accuracy = state.get("best_accuracy", 0.0)
        best_model_path_str = state.get("best_model_path")
        self.best_model_path = Path(best_model_path_str) if best_model_path_str else None
        self.evals_without_improvement = state.get("evals_without_improvement", 0)  # NEW
        
        logger.info(f"✓ Training state loaded:")
        logger.info(f"  - Global step: {self.global_step}")
        logger.info(f"  - Best accuracy: {self.best_accuracy:.2%}")
        logger.info(f"  - Evals without improvement: {self.evals_without_improvement}")
    else:
        logger.warning("⚠️  Training state not found, starting from step 0")
```

### Step 4: 增强 _run_evaluation（保存到文件）
```python
def _run_evaluation(self, is_baseline: bool = False) -> TrainingMetrics:
    """Run evaluation on eval set."""
    # ... 现有的评估代码 ...
    
    # Collect detailed statistics (NEW)
    rubrics = [s.rubric for g in groups for s in g.samples]
    
    eval_stats = {
        "step": self.global_step if not is_baseline else -1,
        "is_baseline": is_baseline,
        "accuracy": accuracy,
        "correct_answers": int(accuracy * len(rubrics)),
        "total_samples": len(rubrics),
        "avg_reward": avg_reward,
        "median_reward": float(np.median(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "min_reward": float(min(rewards)) if rewards else 0.0,
        "max_reward": float(max(rewards)) if rewards else 0.0,
        "avg_score": avg_score,
        "avg_hits": avg_hits,
        "eval_time": eval_time,
    }
    
    # Save to file (NEW)
    if is_baseline:
        eval_log_file = self.output_dir / "baseline_eval.json"
    else:
        eval_log_dir = self.output_dir / "eval_logs"
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        eval_log_file = eval_log_dir / f"eval_step_{self.global_step:04d}.json"
    
    with open(eval_log_file, "w") as f:
        json.dump(eval_stats, f, indent=2)
    
    logger.info(f"💾 Eval stats saved to: {eval_log_file}")
    
    # ... 返回 metrics ...
```

### Step 5: 增强训练循环日志
```python
def train(self):
    """Main training loop."""
    # ... baseline eval ...
    
    for step in range(self.global_step, self.max_steps):
        # ... 
        
        # Collect metrics with more detail
        all_rewards = [s.reward for g in groups for s in g.samples]
        all_scores = [s.rubric.score for g in groups for s in g.samples]
        
        # Calculate additional metrics (NEW)
        median_reward = float(np.median(all_rewards)) if all_rewards else 0.0
        num_early_exit = sum(
            1 for g in groups for s in g.samples 
            if len(s.conversation) < self.policy_config.max_turns * 2
        )
        
        metrics = TrainingMetrics(
            # ... 现有字段 ...
            median_reward=median_reward,  # NEW
            groups_kept=len([g for g in groups if any(s.advantage and s.advantage != 0 for s in g.samples)]),
            groups_filtered=len([g for g in groups if all(s.advantage is None or s.advantage == 0 for s in g.samples)]),
            num_early_exit=num_early_exit,  # NEW
        )
        
        # Print detailed group summary (NEW, if VERBOSE)
        if self.verbose:
            print(f"\n📊 Group Summary:", flush=True)
            print(f"  - Total groups: {len(groups)}", flush=True)
            print(f"  - Groups kept for training: {metrics.groups_kept}", flush=True)
            print(f"  - Groups filtered (low variance): {metrics.groups_filtered}", flush=True)
            print(f"  - Rollouts finished early: {num_early_exit}/{len(all_rewards)}", flush=True)
            print(f"  - Total rollout time: {rollout_time:.1f}s", flush=True)
            print(f"  - Total training time: {train_time:.1f}s", flush=True)
        
        # Enhanced wandb logging (NEW)
        if self.use_wandb:
            wandb.log({
                # ... 现有指标 ...
                "train/median_reward": median_reward,
                "train/groups_kept": metrics.groups_kept,
                "train/groups_filtered": metrics.groups_filtered,
                "train/num_early_exit": num_early_exit,
            }, step=self.global_step)
```

---

## ✅ 验收标准

完成后应该具备以下能力:

1. **Checkpoint 完整性**
   - ✅ checkpoint 目录包含: adapter_model.*, optimizer.pt, training_state.json, training_metadata.json
   - ✅ 断点续训能恢复 optimizer 状态和 early stopping 计数

2. **评估日志**
   - ✅ 每次评估生成 eval_step_XXXX.json
   - ✅ Baseline 评估生成 baseline_eval.json
   - ✅ JSON 包含完整的统计信息

3. **训练日志**
   - ✅ 详细模式下显示 group 统计
   - ✅ 显示 early exit rollout 数量
   - ✅ 显示 trainable token 比例

4. **Wandb 监控**
   - ✅ 包含 median_reward, groups_kept/filtered, early_exit 等指标
   - ✅ 评估指标包含详细的 rubric 统计

5. **用户体验**
   - ✅ RUN_BASELINE_EVAL=false 时自动加载已有 baseline
   - ✅ 断点续训打印详细的恢复信息
   - ✅ 训练结束打印总结信息

---

## 📅 实施时间估计

- **P0 核心功能**: 2-3 小时
  - Checkpoint 增强: 1 小时
  - 评估日志保存: 1 小时
  - Baseline 增强: 0.5 小时

- **P1 日志监控**: 1-2 小时
  - 训练日志增强: 1 小时
  - Wandb 指标补充: 0.5 小时

- **P2 高级功能**: 0.5-1 小时
  - Training summary: 0.5 小时

**总计**: 3.5-6 小时

---

## 🔍 测试计划

1. **功能测试**
   - 从头开始训练 -> 检查生成的文件
   - 中断训练 -> 断点续训 -> 验证状态恢复
   - 运行 baseline -> 关闭 baseline -> 验证从文件加载

2. **兼容性测试**
   - 旧的 checkpoint 能否加载（向后兼容）
   - 缺少 optimizer.pt 时的降级行为

3. **性能测试**
   - 评估日志保存不应显著影响性能
   - 文件大小合理（JSON 不应过大）

---

## 📝 文档更新

完成后需要更新:
1. QUICKSTART_LINKSEARCH.md - 添加新功能使用说明
2. 创建 TRAINING_LOGS.md - 详细说明日志格式和使用方法
3. env.linksearch.example - 添加注释说明 RUN_BASELINE_EVAL

---

## 🎯 预期效果

实施后将实现:
1. ✅ 完整的训练状态保存和恢复
2. ✅ 可追溯的评估历史记录
3. ✅ 更清晰的训练进度监控
4. ✅ 更好的调试和问题定位能力
5. ✅ 与 rl-unsloth 项目的功能对齐

