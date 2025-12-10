"""Custom GRPO Trainer for Link Search Agent with token-level masking."""

import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from link_search_agent.config import PolicyConfig
from link_search_agent.data.types import LinkSearchQuery
from link_search_agent.rollout import LinkSearchRubric, calculate_reward, normalize_handle
from link_search_agent.grpo_utils import execute_rollout
from link_search_agent.rollout_logger import save_rollout_logs

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySample:
    """Single trajectory collected from a rollout."""
    query_id: str
    query: LinkSearchQuery
    conversation: List[Dict]
    reward: float
    rubric: LinkSearchRubric
    rollout_idx: int
    group_id: int
    advantage: Optional[float] = None
    turn_advantages: Optional[List[float]] = None
    rollout_log: Optional[object] = None


@dataclass
class TrajectoryGroup:
    """Grouped trajectories for GRPO."""
    query: LinkSearchQuery
    group_id: int
    samples: List[TrajectorySample] = field(default_factory=list)


@dataclass
class TokenizedTrajectory:
    """Tokenized trajectory ready for batching."""
    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    attention_mask: torch.Tensor
    advantage_mask: torch.Tensor
    group_id: int
    query_id: str
    old_logprobs: Optional[torch.Tensor] = None


@dataclass
class TrainingMetrics:
    """Metrics for a training step."""
    loss: float
    policy_loss: float
    kl_loss: float
    avg_reward: float
    max_reward: float
    min_reward: float
    accuracy: float
    avg_score: float = 0.0
    avg_hits: float = 0.0
    num_trainable_tokens: int = 0
    num_total_tokens: int = 0
    rollout_time: float = 0.0
    training_time: float = 0.0
    reward_std: float = 0.0
    groups_kept: int = 0
    groups_filtered: int = 0


class LinkSearchGRPOTrainer:
    """Custom GRPO Trainer for Link Search Agent."""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_queries: List[LinkSearchQuery],
        eval_queries: List[LinkSearchQuery],
        policy_config: PolicyConfig,
        num_rollouts: int = 4,
        learning_rate: float = 1e-5,
        beta: float = 0.01,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "outputs/grpo_linksearch",
        target_accuracy: float = 0.80,
        eval_steps: int = 10,
        save_steps: int = 10,
        max_steps: int = 1000,
        warmup_steps: int = 10,
        patience: int = 5,
        min_group_std: float = 0.05,
        resume_from_checkpoint: Optional[str] = None,
        clip_epsilon: float = 0.2,
        rollout_concurrency: int = 4,
        eval_rollouts: int = 1,
        max_seq_length: Optional[int] = None,
        use_wandb: bool = False,
        run_baseline_eval: bool = True,
        enable_detailed_logging: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_queries = train_queries
        self.eval_queries = eval_queries
        self.policy_config = policy_config
        self.num_rollouts = num_rollouts
        self.beta = beta
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.target_accuracy = target_accuracy
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.min_group_std = min_group_std
        self.verbose = policy_config.verbose
        self.clip_epsilon = clip_epsilon
        self.rollout_concurrency = max(1, rollout_concurrency)
        self.eval_rollouts = max(1, eval_rollouts)
        self.max_seq_length = max_seq_length or getattr(
            getattr(self.model, "config", {}), "max_position_embeddings", 4096
        )
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.run_baseline_eval = run_baseline_eval
        self.enable_detailed_logging = enable_detailed_logging
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Reference model state for KL
        self.ref_model_state = None
        if beta > 0:
            logger.info("Saving reference model state for KL divergence...")
            self.ref_model_state = {
                name: param.detach().cpu().clone() 
                for name, param in model.named_parameters()
            }
        
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.evals_without_improvement = 0
        
        # Resume if checkpoint provided
        if resume_from_checkpoint:
            self._load_checkpoint(Path(resume_from_checkpoint))
        
        logger.info("="*60)
        logger.info("LinkSearchGRPOTrainer initialized")
        logger.info("="*60)
        logger.info(f"Train queries: {len(train_queries)}")
        logger.info(f"Eval queries: {len(eval_queries)}")
        logger.info(f"Rollouts per query: {num_rollouts}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"KL penalty (beta): {beta}")
        logger.info(f"Target accuracy: {target_accuracy*100:.1f}%")
        logger.info("="*60)
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load training state from checkpoint."""
        logger.info(f"📂 Loading training state from checkpoint: {checkpoint_path}")
        
        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            logger.info("✓ Optimizer state loaded")
        else:
            logger.warning("⚠️  Optimizer state not found, starting with fresh optimizer")
        
        # Load training state
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.global_step = state.get("global_step", 0)
            self.best_accuracy = state.get("best_accuracy", 0.0)
            best_model_path_str = state.get("best_model_path")
            self.best_model_path = Path(best_model_path_str) if best_model_path_str else None
            self.evals_without_improvement = state.get("evals_without_improvement", 0)
            
            logger.info(f"✓ Training state loaded:")
            logger.info(f"  - Global step: {self.global_step}")
            logger.info(f"  - Best accuracy: {self.best_accuracy:.2%}")
            logger.info(f"  - Evals without improvement: {self.evals_without_improvement}")
        else:
            logger.warning("⚠️  Training state not found, starting from step 0")
    
    def _save_checkpoint(self, metrics: TrainingMetrics):
        """Save model checkpoint and training state."""
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "best_accuracy": self.best_accuracy,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "evals_without_improvement": self.evals_without_improvement,
        }
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # Save training metadata with detailed metrics
        training_metadata = {
            "step": self.global_step,
            "accuracy": metrics.accuracy,
            "metrics": {
                "loss": metrics.loss,
                "policy_loss": metrics.policy_loss,
                "kl_loss": metrics.kl_loss,
                "avg_reward": metrics.avg_reward,
                "max_reward": metrics.max_reward,
                "min_reward": metrics.min_reward,
                "avg_score": metrics.avg_score,
                "avg_hits": metrics.avg_hits,
                "reward_std": getattr(metrics, 'reward_std', 0.0),
                "rollout_time": metrics.rollout_time,
                "training_time": metrics.training_time,
                "groups_kept": metrics.groups_kept,
                "groups_filtered": metrics.groups_filtered,
            }
        }
        with open(checkpoint_dir / "training_metadata.json", 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        logger.info(f"💾 Model and training state saved to: {checkpoint_dir}")
        return checkpoint_dir
    
    async def _collect_rollouts(
        self,
        queries: List[LinkSearchQuery],
        is_eval: bool = False,
    ) -> Tuple[List[TrajectoryGroup], Dict[str, float]]:
        """Collect rollouts for a batch of queries.
        
        Returns:
            Tuple of (groups, timing_dict) where timing_dict contains:
            - total_time_ms: Total time for all rollouts
            - avg_query_time_ms: Average time per query
            - avg_group_time_ms: Average time per group
        """
        groups = []
        all_rollout_logs = []
        
        num_rollouts = 1 if is_eval else self.num_rollouts
        
        collection_start = time.time()
        group_times = []
        query_times = []
        
        for group_id, query in enumerate(queries):
            group_start = time.time()
            group = TrajectoryGroup(query=query, group_id=group_id)
            
            for rollout_idx in range(num_rollouts):
                try:
                    query_start = time.time()
                    conversation, reward, rubric, rollout_log = await execute_rollout(
                        query=query,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        policy_config=self.policy_config,
                        verbose=self.verbose,
                        log_turns=self.verbose,
                        rollout_index=rollout_idx,
                        num_rollouts=num_rollouts,
                        enable_detailed_logging=self.enable_detailed_logging,
                        training_step=self.global_step,
                    )
                    query_time_ms = (time.time() - query_start) * 1000.0
                    query_times.append(query_time_ms)
                    
                    sample = TrajectorySample(
                        query_id=query.id,
                        query=query,
                        conversation=conversation,
                        reward=reward,
                        rubric=rubric,
                        rollout_idx=rollout_idx,
                        group_id=group_id,
                        rollout_log=rollout_log,
                    )
                    group.samples.append(sample)
                    
                    if rollout_log:
                        all_rollout_logs.append(rollout_log)
                    
                except Exception as e:
                    logger.error(f"Rollout failed for query {query.id}: {e}")
                    continue
            
            group_time_ms = (time.time() - group_start) * 1000.0
            group_times.append(group_time_ms)
            
            if group.samples:
                groups.append(group)
        
        total_time_ms = (time.time() - collection_start) * 1000.0
        
        timing_dict = {
            "total_time_ms": total_time_ms,
            "avg_query_time_ms": np.mean(query_times) if query_times else 0.0,
            "avg_group_time_ms": np.mean(group_times) if group_times else 0.0,
            "min_query_time_ms": np.min(query_times) if query_times else 0.0,
            "max_query_time_ms": np.max(query_times) if query_times else 0.0,
            "min_group_time_ms": np.min(group_times) if group_times else 0.0,
            "max_group_time_ms": np.max(group_times) if group_times else 0.0,
        }
        
        # Save rollout logs
        if self.enable_detailed_logging and all_rollout_logs:
            save_rollout_logs(all_rollout_logs, str(self.output_dir / "rollout_logs"))
        
        return groups, timing_dict
    
    def _compute_advantages(self, groups: List[TrajectoryGroup]) -> List[TrajectoryGroup]:
        """Compute GRPO advantages for each group."""
        for group in groups:
            rewards = [s.reward for s in group.samples]
            
            if len(rewards) < 2:
                for s in group.samples:
                    s.advantage = 0.0
                continue
            
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            
            # Filter low-variance groups
            if std_reward < self.min_group_std:
                for s in group.samples:
                    s.advantage = 0.0
                continue
            
            # Normalize advantages
            for s in group.samples:
                s.advantage = (s.reward - mean_reward) / (std_reward + 1e-8)
        
        return groups
    
    def _compute_action_advantage(
        self,
        msg: Dict,
        rubric: LinkSearchRubric,
        query: LinkSearchQuery,
        msg_idx: int,
        conversation: List[Dict],
        trajectory_advantage: float,
    ) -> float:
        """Compute action-level advantage for process rewards."""
        tool_calls = msg.get("tool_calls", [])
        
        if not tool_calls:
            return trajectory_advantage
        
        next_msg = conversation[msg_idx + 1] if msg_idx + 1 < len(conversation) else None
        gold_set = set(normalize_handle(h) for h in query.gold_handles)
        
        for tc in tool_calls:
            func_name = tc.get("function", {}).get("name")
            func_args_str = tc.get("function", {}).get("arguments", "{}")
            
            try:
                func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
            except:
                return -2.0
            
            if func_name == "search_profile":
                if next_msg and next_msg.get("role") == "tool":
                    try:
                        tool_result = json.loads(next_msg.get("content", "{}"))
                        if isinstance(tool_result, dict) and tool_result.get("success"):
                            rows = tool_result.get("rows", [])
                            
                            # Check if found correct handles
                            found_correct = False
                            for row in rows:
                                handle = row.get("handle") or row.get("linkedin_handle", "")
                                if normalize_handle(handle) in gold_set:
                                    found_correct = True
                                    break
                            
                            if found_correct:
                                return +1.0  # Found correct profile
                            elif len(rows) > 0:
                                return +0.1  # Found something
                            else:
                                return -0.2  # Zero results
                    except:
                        pass
                return trajectory_advantage * 0.5
            
            elif func_name == "read_profile":
                handle = func_args.get("linkedin_handle", "").lower()
                if normalize_handle(handle) in gold_set:
                    return +1.2  # Read correct profile
                else:
                    return -0.3  # Read wrong profile
            
            elif func_name == "return_results":
                results = func_args.get("results", {})
                if isinstance(results, dict):
                    num_correct = 0
                    for h in results.keys():
                        if normalize_handle(h) in gold_set:
                            num_correct += 1
                    
                    if num_correct > 0:
                        return +1.5 + 0.2 * num_correct
                    else:
                        return -0.5
                return trajectory_advantage
            
            else:
                return -2.0
        
        return trajectory_advantage * 0.5
    
    def tokenize_conversation_with_mask(
        self,
        conversation: List[Dict],
        rubric: Optional[LinkSearchRubric] = None,
        query: Optional[LinkSearchQuery] = None,
        trajectory_advantage: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize conversation and create loss mask with advantages."""
        all_tokens = []
        all_masks = []
        all_advantages = []
        
        for msg_idx, msg in enumerate(conversation):
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            
            is_model_generated = (role == "assistant")
            
            # Serialize message
            if role == "system":
                text = f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text = f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                if tool_calls:
                    tool_calls_str = json.dumps(tool_calls)
                    text = f"<|im_start|>assistant\n{tool_calls_str}<|im_end|>\n"
                else:
                    text = f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                text = f"<|im_start|>tool\ntool_call_id: {tool_call_id}\n{content}<|im_end|>\n"
            else:
                text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if is_model_generated:
                mask = [1.0] * len(tokens)
                
                if rubric is not None and query is not None:
                    action_advantage = self._compute_action_advantage(
                        msg, rubric, query, msg_idx, conversation, trajectory_advantage
                    )
                else:
                    action_advantage = trajectory_advantage
                
                advantages = [action_advantage] * len(tokens)
            else:
                mask = [0.0] * len(tokens)
                advantages = [0.0] * len(tokens)
            
            all_tokens.extend(tokens)
            all_masks.extend(mask)
            all_advantages.extend(advantages)
        
        # Add EOS
        all_tokens.append(self.tokenizer.eos_token_id)
        all_masks.append(0.0)
        all_advantages.append(0.0)
        
        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        labels = input_ids.clone()
        loss_mask = torch.tensor(all_masks, dtype=torch.float)
        advantage_mask = torch.tensor(all_advantages, dtype=torch.float)
        
        return input_ids, labels, loss_mask, advantage_mask
    
    def compute_loss_for_trajectory(
        self,
        conversation: List[Dict],
        advantage: float,
        rubric: Optional[LinkSearchRubric] = None,
        query: Optional[LinkSearchQuery] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a single trajectory."""
        input_ids, labels, loss_mask, advantage_mask = self.tokenize_conversation_with_mask(
            conversation, rubric, query, advantage
        )
        
        device = next(self.model.parameters()).device
        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        loss_mask = loss_mask.to(device)
        advantage_mask = advantage_mask.to(device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = loss_mask[1:].contiguous()
        shift_advantage = advantage_mask[1:].contiguous()
        
        # Compute losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        masked_losses = token_losses * shift_mask * shift_advantage
        
        num_trainable = shift_mask.sum().item()
        
        if num_trainable == 0:
            policy_loss = torch.tensor(0.0, device=device)
        else:
            policy_loss = masked_losses.sum() / shift_mask.sum()
        
        # KL divergence (simplified - no ref model copy)
        kl_loss = torch.tensor(0.0, device=device)
        
        total_loss = policy_loss + kl_loss
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "num_trainable_tokens": num_trainable,
            "num_total_tokens": shift_mask.numel(),
        }
        
        return total_loss, metrics
    
    def _run_evaluation(self, is_baseline: bool = False) -> TrainingMetrics:
        """Run evaluation on eval set."""
        eval_type = "BASELINE EVALUATION" if is_baseline else "EVALUATION"
        num_eval_samples = min(50, len(self.eval_queries))
        
        print("", flush=True)
        print("="*80, flush=True)
        print(f"🔍 {eval_type} - Step {self.global_step}", flush=True)
        print("="*80, flush=True)
        print(f"Evaluating on {num_eval_samples} queries...", flush=True)
        if not is_baseline:
            print(f"Current best accuracy: {self.best_accuracy:.2%}", flush=True)
            print(f"Evaluations without improvement: {self.evals_without_improvement}/{self.patience}", flush=True)
        print("="*80, flush=True)
        
        logger.info(f"Running {eval_type.lower()} on {num_eval_samples} queries...")
        
        self.model.eval()
        
        eval_start = time.time()
        
        loop = asyncio.get_event_loop()
        groups, eval_timing = loop.run_until_complete(
            self._collect_rollouts(self.eval_queries[:num_eval_samples], is_eval=True)
        )
        
        rewards = []
        scores = []
        hits = []
        rubrics = []
        
        for group in groups:
            for sample in group.samples:
                rewards.append(sample.reward)
                scores.append(sample.rubric.score)
                hits.append(sample.rubric.num_correct_handles)
                rubrics.append(sample.rubric)
        
        avg_reward = np.mean(rewards) if rewards else 0.0
        avg_score = np.mean(scores) if scores else 0.0
        avg_hits = np.mean(hits) if hits else 0.0
        median_reward = float(np.median(rewards)) if rewards else 0.0
        std_reward = float(np.std(rewards)) if rewards else 0.0
        
        # Accuracy = percentage with score > 0.5
        accuracy = np.mean([1 if s > 0.5 else 0 for s in scores]) if scores else 0.0
        correct_answers = int(accuracy * len(scores)) if scores else 0
        
        # Collect detailed rubric statistics
        attempted_answer = sum(1 for r in rubrics if r.num_correct_handles > 0 or r.score > 0)
        found_correct_profile = sum(1 for r in rubrics if r.num_correct_handles > 0)
        
        # Calculate average turns for different outcomes
        turns_list = [len(s.conversation) // 2 for g in groups for s in g.samples]
        avg_turns = np.mean(turns_list) if turns_list else 0.0
        
        eval_time = time.time() - eval_start
        
        # Create evaluation statistics dictionary with detailed timing
        eval_stats = {
            "step": self.global_step if not is_baseline else -1,
            "is_baseline": is_baseline,
            "accuracy": float(accuracy),
            "correct_answers": correct_answers,
            "total_samples": len(rubrics),
            "attempted_answer": attempted_answer,
            "avg_reward": float(avg_reward),
            "median_reward": median_reward,
            "std_reward": std_reward,
            "min_reward": float(min(rewards)) if rewards else 0.0,
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "avg_score": float(avg_score),
            "avg_hits": float(avg_hits),
            "found_correct_profile": found_correct_profile,
            "avg_turns": float(avg_turns),
            "eval_time_seconds": float(eval_time),
            "eval_time_ms": float(eval_time * 1000.0),
            "avg_query_time_ms": eval_timing.get("avg_query_time_ms", 0.0),
            "min_query_time_ms": eval_timing.get("min_query_time_ms", 0.0),
            "max_query_time_ms": eval_timing.get("max_query_time_ms", 0.0),
        }
        
        # Save to file
        if is_baseline:
            eval_log_file = self.output_dir / "baseline_eval.json"
        else:
            eval_log_dir = self.output_dir / "eval_logs"
            eval_log_dir.mkdir(parents=True, exist_ok=True)
            eval_log_file = eval_log_dir / f"eval_step_{self.global_step:04d}.json"
        
        with open(eval_log_file, "w") as f:
            json.dump(eval_stats, f, indent=2)
        
        # Print evaluation results with detailed timing
        print("", flush=True)
        print(f"📊 {eval_type} Results:", flush=True)
        print(f"  Accuracy: {accuracy:.2%} ({correct_answers}/{len(rubrics)})", flush=True)
        print(f"  Avg Score: {avg_score:.3f}", flush=True)
        print(f"  Avg Hits: {avg_hits:.2f}", flush=True)
        print(f"  Avg Reward: {avg_reward:.3f} (median: {median_reward:.3f}, std: {std_reward:.3f})", flush=True)
        print(f"  Found Correct Profile: {found_correct_profile}/{len(rubrics)} ({found_correct_profile/max(len(rubrics), 1)*100:.1f}%)", flush=True)
        print(f"  Avg Turns: {avg_turns:.2f}", flush=True)
        print(f"\n⏱️  Timing Breakdown:", flush=True)
        print(f"  Total Eval Time: {eval_time:.2f}s ({eval_time*1000:.0f}ms)", flush=True)
        print(f"  Avg Query Time: {eval_timing.get('avg_query_time_ms', 0):.2f}ms", flush=True)
        print(f"  Query Time Range: {eval_timing.get('min_query_time_ms', 0):.2f}ms - {eval_timing.get('max_query_time_ms', 0):.2f}ms", flush=True)
        print(f"💾 Eval stats saved to: {eval_log_file}", flush=True)
        print("="*80, flush=True)
        print("", flush=True)
        
        self.model.train()
        
        return TrainingMetrics(
            loss=0.0,
            policy_loss=0.0,
            kl_loss=0.0,
            avg_reward=avg_reward,
            max_reward=max(rewards) if rewards else 0.0,
            min_reward=min(rewards) if rewards else 0.0,
            accuracy=accuracy,
            avg_score=avg_score,
            avg_hits=avg_hits,
        )
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Handle baseline evaluation
        baseline_file = self.output_dir / "baseline_eval.json"
        if self.global_step == 0:
            if self.run_baseline_eval:
                # Run new baseline evaluation
                print("\n" + "="*80, flush=True)
                print("🏁 Running Baseline Evaluation (before training)", flush=True)
                print("="*80, flush=True)
                baseline_metrics = self._run_evaluation(is_baseline=True)
                print(f"📊 Baseline Performance:", flush=True)
                print(f"  Accuracy: {baseline_metrics.accuracy:.2%}", flush=True)
                print(f"  Avg Score: {baseline_metrics.avg_score:.3f}", flush=True)
                print(f"  Avg Hits: {baseline_metrics.avg_hits:.2f}", flush=True)
                print("="*80, flush=True)
                print("", flush=True)
                
                if self.use_wandb:
                    wandb.log({
                        "baseline/accuracy": baseline_metrics.accuracy,
                        "baseline/avg_score": baseline_metrics.avg_score,
                        "baseline/avg_hits": baseline_metrics.avg_hits,
                        "baseline/avg_reward": baseline_metrics.avg_reward,
                    }, step=0)
            elif baseline_file.exists():
                # Load baseline from file
                print("\n" + "="*80, flush=True)
                print("📂 Loading Baseline Evaluation from file", flush=True)
                print("="*80, flush=True)
                try:
                    with open(baseline_file, 'r') as f:
                        baseline_stats = json.load(f)
                    
                    print(f"✓ Loaded baseline from: {baseline_file}", flush=True)
                    print(f"📊 Baseline Performance:", flush=True)
                    print(f"  Accuracy: {baseline_stats.get('accuracy', 0):.2%}", flush=True)
                    print(f"  Avg Score: {baseline_stats.get('avg_score', 0):.3f}", flush=True)
                    print(f"  Avg Hits: {baseline_stats.get('avg_hits', 0):.2f}", flush=True)
                    print(f"  Avg Reward: {baseline_stats.get('avg_reward', 0):.3f}", flush=True)
                    print("="*80, flush=True)
                    print("", flush=True)
                    
                    if self.use_wandb:
                        wandb.log({
                            "baseline/accuracy": baseline_stats.get('accuracy', 0),
                            "baseline/avg_score": baseline_stats.get('avg_score', 0),
                            "baseline/avg_hits": baseline_stats.get('avg_hits', 0),
                            "baseline/avg_reward": baseline_stats.get('avg_reward', 0),
                        }, step=0)
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load baseline from file: {e}")
                    print(f"⚠️  Failed to load baseline: {e}", flush=True)
            else:
                print("\n⚠️  Baseline evaluation skipped (RUN_BASELINE_EVAL=false and no baseline file found)", flush=True)
                print(f"   To run baseline: set RUN_BASELINE_EVAL=true or provide {baseline_file}", flush=True)
                print("", flush=True)
        
        self.model.train()
        
        query_idx = 0
        
        for step in range(self.global_step, self.max_steps):
            self.global_step = step + 1
            step_start = time.time()
            
            # Get batch of queries
            batch_queries = []
            for _ in range(self.batch_size):
                batch_queries.append(self.train_queries[query_idx % len(self.train_queries)])
                query_idx += 1
            
            # Collect rollouts
            rollout_start = time.time()
            loop = asyncio.get_event_loop()
            groups, rollout_timing = loop.run_until_complete(self._collect_rollouts(batch_queries))
            rollout_time = time.time() - rollout_start
            
            # Compute advantages
            advantage_start = time.time()
            groups = self._compute_advantages(groups)
            advantage_time = time.time() - advantage_start
            
            # Training step
            train_start = time.time()
            
            total_loss = 0.0
            total_policy_loss = 0.0
            num_samples = 0
            
            self.optimizer.zero_grad()
            
            for group in groups:
                for sample in group.samples:
                    if sample.advantage is None or sample.advantage == 0:
                        continue
                    
                    loss, metrics = self.compute_loss_for_trajectory(
                        sample.conversation,
                        sample.advantage,
                        sample.rubric,
                        sample.query,
                    )
                    
                    if loss.item() != 0:
                        (loss / self.gradient_accumulation_steps).backward()
                        total_loss += loss.item()
                        total_policy_loss += metrics["policy_loss"]
                        num_samples += 1
            
            if num_samples > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            train_time = time.time() - train_start
            step_total_time = time.time() - step_start
            
            # Collect metrics
            all_rewards = [s.reward for g in groups for s in g.samples]
            all_scores = [s.rubric.score for g in groups for s in g.samples]
            
            # Calculate additional statistics
            median_reward = float(np.median(all_rewards)) if all_rewards else 0.0
            reward_std = float(np.std(all_rewards)) if all_rewards else 0.0
            
            # Count early exits (rollouts that finished before max turns)
            num_early_exit = sum(
                1 for g in groups for s in g.samples 
                if len(s.conversation) < self.policy_config.max_turns * 2
            )
            
            groups_kept = sum(1 for g in groups if any(s.advantage and s.advantage != 0 for s in g.samples))
            groups_filtered = sum(1 for g in groups if all(s.advantage is None or s.advantage == 0 for s in g.samples))
            
            metrics = TrainingMetrics(
                loss=total_loss / max(num_samples, 1),
                policy_loss=total_policy_loss / max(num_samples, 1),
                kl_loss=0.0,
                avg_reward=np.mean(all_rewards) if all_rewards else 0.0,
                max_reward=max(all_rewards) if all_rewards else 0.0,
                min_reward=min(all_rewards) if all_rewards else 0.0,
                accuracy=np.mean([1 if s > 0.5 else 0 for s in all_scores]) if all_scores else 0.0,
                avg_score=np.mean(all_scores) if all_scores else 0.0,
                rollout_time=rollout_time,
                training_time=train_time,
                reward_std=reward_std,
                groups_kept=groups_kept,
                groups_filtered=groups_filtered,
            )
            
            # Log progress (simple mode)
            if self.verbose:
                # Detailed mode
                print("", flush=True)
                print("="*80, flush=True)
                print(f"📍 STEP {self.global_step}/{self.max_steps}", flush=True)
                print("="*80, flush=True)
                print(f"Loss: {metrics.loss:.4f} | Score: {metrics.avg_score:.3f} | Reward: {metrics.avg_reward:.3f} (median: {median_reward:.3f})", flush=True)
                print(f"📊 Group Summary:", flush=True)
                print(f"  - Total groups: {len(groups)}", flush=True)
                print(f"  - Groups kept for training: {groups_kept}", flush=True)
                print(f"  - Groups filtered (low variance): {groups_filtered}", flush=True)
                print(f"  - Rollouts finished early: {num_early_exit}/{len(all_rewards)}", flush=True)
                print(f"\n⏱️  Timing Breakdown (Step {self.global_step}):", flush=True)
                print(f"  - Rollout collection: {rollout_time:.2f}s ({rollout_time*1000:.0f}ms)", flush=True)
                print(f"    • Avg query time: {rollout_timing.get('avg_query_time_ms', 0):.2f}ms", flush=True)
                print(f"    • Avg group time: {rollout_timing.get('avg_group_time_ms', 0):.2f}ms", flush=True)
                print(f"    • Query time range: {rollout_timing.get('min_query_time_ms', 0):.0f}-{rollout_timing.get('max_query_time_ms', 0):.0f}ms", flush=True)
                print(f"  - Advantage computation: {advantage_time:.2f}s ({advantage_time*1000:.0f}ms)", flush=True)
                print(f"  - Training step: {train_time:.2f}s ({train_time*1000:.0f}ms)", flush=True)
                print(f"  - Step total: {step_total_time:.2f}s ({step_total_time*1000:.0f}ms)", flush=True)
                print("="*80, flush=True)
            else:
                # Simple mode with timing
                print(
                    f"Step {self.global_step}/{self.max_steps} | "
                    f"Loss: {metrics.loss:.4f} | "
                    f"Score: {metrics.avg_score:.3f} | "
                    f"Reward: {metrics.avg_reward:.3f} | "
                    f"Groups: {groups_kept}/{len(groups)} | "
                    f"Rollout: {rollout_time:.1f}s | "
                    f"Train: {train_time:.1f}s | "
                    f"Total: {step_total_time:.1f}s",
                    flush=True
                )
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "train/loss": metrics.loss,
                    "train/policy_loss": metrics.policy_loss,
                    "train/avg_reward": metrics.avg_reward,
                    "train/median_reward": median_reward,
                    "train/reward_std": reward_std,
                    "train/avg_score": metrics.avg_score,
                    "train/accuracy": metrics.accuracy,
                    "train/rollout_time": rollout_time,
                    "train/training_time": train_time,
                    "train/groups_kept": groups_kept,
                    "train/groups_filtered": groups_filtered,
                    "train/num_early_exit": num_early_exit,
                }, step=self.global_step)
            
            # Evaluation
            if self.global_step % self.eval_steps == 0:
                eval_metrics = self._run_evaluation(is_baseline=False)
                
                # Load detailed eval stats for wandb logging
                eval_log_file = self.output_dir / "eval_logs" / f"eval_step_{self.global_step:04d}.json"
                if self.use_wandb and eval_log_file.exists():
                    try:
                        with open(eval_log_file, 'r') as f:
                            eval_stats = json.load(f)
                        
                        wandb.log({
                            "eval/avg_score": eval_metrics.avg_score,
                            "eval/accuracy": eval_metrics.accuracy,
                            "eval/avg_reward": eval_metrics.avg_reward,
                            "eval/median_reward": eval_stats.get("median_reward", 0.0),
                            "eval/std_reward": eval_stats.get("std_reward", 0.0),
                            "eval/min_reward": eval_stats.get("min_reward", 0.0),
                            "eval/max_reward": eval_stats.get("max_reward", 0.0),
                            "eval/avg_hits": eval_metrics.avg_hits,
                            "eval/attempted_answer": eval_stats.get("attempted_answer", 0),
                            "eval/found_profile_rate": eval_stats.get("found_correct_profile", 0) / max(eval_stats.get("total_samples", 1), 1),
                            "eval/avg_turns": eval_stats.get("avg_turns", 0.0),
                        }, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to load eval stats for wandb: {e}")
                        # Fallback to basic logging
                        wandb.log({
                            "eval/avg_score": eval_metrics.avg_score,
                            "eval/accuracy": eval_metrics.accuracy,
                            "eval/avg_reward": eval_metrics.avg_reward,
                            "eval/avg_hits": eval_metrics.avg_hits,
                        }, step=self.global_step)
                elif self.use_wandb:
                    # Fallback to basic logging
                    wandb.log({
                        "eval/avg_score": eval_metrics.avg_score,
                        "eval/accuracy": eval_metrics.accuracy,
                        "eval/avg_reward": eval_metrics.avg_reward,
                        "eval/avg_hits": eval_metrics.avg_hits,
                    }, step=self.global_step)
                
                # Check for improvement
                if eval_metrics.accuracy > self.best_accuracy:
                    self.best_accuracy = eval_metrics.accuracy
                    self.evals_without_improvement = 0
                    self.best_model_path = self._save_checkpoint(eval_metrics)
                    print(f"🎯 New best accuracy: {self.best_accuracy:.2%}", flush=True)
                else:
                    self.evals_without_improvement += 1
                
                # Early stopping
                if self.evals_without_improvement >= self.patience:
                    print(f"Early stopping: no improvement for {self.patience} evaluations", flush=True)
                    break
                
                # Target reached
                if eval_metrics.accuracy >= self.target_accuracy:
                    print(f"🎉 Target accuracy {self.target_accuracy:.2%} reached!", flush=True)
                    self._save_checkpoint(eval_metrics)
                    break
            
            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint(metrics)
        
        # Calculate training time
        total_training_time = time.time() - step_start if 'step_start' in locals() else 0
        
        # Final save
        final_dir = self.output_dir / "final"
        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        
        # Print comprehensive training summary
        print("", flush=True)
        print("="*80, flush=True)
        print("🎉 TRAINING SUMMARY", flush=True)
        print("="*80, flush=True)
        print(f"Total steps completed: {self.global_step}/{self.max_steps}", flush=True)
        print(f"Best accuracy achieved: {self.best_accuracy:.2%}", flush=True)
        print(f"Target accuracy: {self.target_accuracy:.2%}", flush=True)
        
        if self.best_accuracy >= self.target_accuracy:
            print(f"✅ Target accuracy REACHED!", flush=True)
        else:
            print(f"⚠️  Target accuracy not reached (stopped early or max steps)", flush=True)
        
        if self.evals_without_improvement >= self.patience:
            print(f"🛑 Stopped due to early stopping (no improvement for {self.patience} evaluations)", flush=True)
        elif self.global_step >= self.max_steps:
            print(f"🏁 Stopped: reached max steps", flush=True)
        elif self.best_accuracy >= self.target_accuracy:
            print(f"🎯 Stopped: target accuracy reached", flush=True)
        
        print("", flush=True)
        print("📁 Model Locations:", flush=True)
        print(f"  - Final model: {final_dir}", flush=True)
        if self.best_model_path:
            print(f"  - Best model (accuracy: {self.best_accuracy:.2%}): {self.best_model_path}", flush=True)
        
        # Count total checkpoints saved
        checkpoints = list(self.output_dir.glob("checkpoint-*"))
        print(f"  - Total checkpoints saved: {len(checkpoints)}", flush=True)
        
        # Check for eval logs
        eval_logs_dir = self.output_dir / "eval_logs"
        if eval_logs_dir.exists():
            eval_logs = list(eval_logs_dir.glob("eval_step_*.json"))
            print(f"  - Evaluation logs: {len(eval_logs)} files in {eval_logs_dir}", flush=True)
        
        print("="*80, flush=True)
        print("✅ Training complete!", flush=True)
        print("="*80, flush=True)
        print("", flush=True)
        
        logger.info("="*80)
        logger.info("Training Summary")
        logger.info("="*80)
        logger.info(f"Total steps: {self.global_step}/{self.max_steps}")
        logger.info(f"Best accuracy: {self.best_accuracy:.2%}")
        logger.info(f"Target accuracy: {self.target_accuracy:.2%}")
        if self.best_model_path:
            logger.info(f"Best model: {self.best_model_path}")
        logger.info(f"Final model: {final_dir}")
        logger.info("="*80)

