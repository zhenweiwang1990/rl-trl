#!/usr/bin/env python3
"""
Evaluation script for Link Search Agent trained models.

This script evaluates a trained model on the Link Search task.

Usage:
    # Evaluate final model
    python eval_linksearch.py --checkpoint outputs/grpo_linksearch_masked/final
    
    # Evaluate specific checkpoint
    python eval_linksearch.py --checkpoint outputs/grpo_linksearch_masked/checkpoint-0100
    
    # Evaluate best model with custom eval size
    python eval_linksearch.py --checkpoint outputs/grpo_linksearch_masked/best_model --eval-size 200
    
    # Save detailed results
    python eval_linksearch.py --checkpoint outputs/grpo_linksearch_masked/final --save-results
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from unsloth import FastLanguageModel

# Load environment variables
try:
    from dotenv import load_dotenv
    env_file = Path(".env")
    if not env_file.exists():
        env_file = Path("env.linksearch")
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment variables from {env_file}")
except ImportError:
    print("⚠ python-dotenv not installed, using system environment variables")

# Local imports
from link_search_agent.config import PolicyConfig
from link_search_agent.data import load_link_search_queries
from link_search_agent.grpo_utils import execute_rollout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint_path: str, max_seq_length: int = 4096):
    """Load trained model and tokenizer."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            dtype=None,
            device_map="auto",
        )
        
        # Enable inference mode and optimizations
        FastLanguageModel.for_inference(model)
        model.eval()  # Ensure model is in eval mode
        
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        logger.info("✓ Model loaded successfully with inference optimizations")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


async def evaluate_model(
    model,
    tokenizer,
    queries: List,
    policy_config: PolicyConfig,
    num_rollouts: int = 1,
    verbose: bool = False,
) -> Dict:
    """Evaluate model on queries."""
    logger.info(f"Evaluating on {len(queries)} queries with {num_rollouts} rollout(s) each...")
    
    eval_start = time.time()
    all_rewards = []
    all_scores = []
    all_hits = []
    all_rubrics = []
    all_turns = []
    
    for i, query in enumerate(queries):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Progress: {i+1}/{len(queries)} queries evaluated")
        
        query_rewards = []
        query_scores = []
        
        for rollout_idx in range(num_rollouts):
            try:
                conversation, reward, rubric, rollout_log = await execute_rollout(
                    query=query,
                    model=model,
                    tokenizer=tokenizer,
                    policy_config=policy_config,
                    verbose=verbose,
                    log_turns=verbose,
                    rollout_index=rollout_idx,
                    num_rollouts=num_rollouts,
                    enable_detailed_logging=False,
                    training_step=-1,  # Evaluation mode
                )
                
                query_rewards.append(reward)
                query_scores.append(rubric.score)
                all_hits.append(rubric.num_correct_handles)
                all_rubrics.append(rubric)
                all_turns.append(len(conversation) // 2)
                
                if verbose and rollout_idx == 0:
                    logger.info(f"\n  Query {i+1}: {query.question[:80]}...")
                    logger.info(f"  Score: {rubric.score:.3f}, Reward: {reward:.3f}, Hits: {rubric.num_correct_handles}")
                
            except Exception as e:
                logger.error(f"Error evaluating query {i+1}, rollout {rollout_idx}: {e}")
                query_rewards.append(0.0)
                query_scores.append(0.0)
        
        # Average across rollouts for this query
        all_rewards.append(np.mean(query_rewards))
        all_scores.append(np.mean(query_scores))
    
    eval_time = time.time() - eval_start
    
    # Calculate statistics
    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    median_reward = float(np.median(all_rewards)) if all_rewards else 0.0
    std_reward = float(np.std(all_rewards)) if all_rewards else 0.0
    avg_score = float(np.mean(all_scores)) if all_scores else 0.0
    avg_hits = float(np.mean(all_hits)) if all_hits else 0.0
    avg_turns = float(np.mean(all_turns)) if all_turns else 0.0
    
    # Accuracy = percentage with score > 0.5
    accuracy = float(np.mean([1 if s > 0.5 else 0 for s in all_scores])) if all_scores else 0.0
    correct_answers = int(accuracy * len(all_scores))
    
    # Detailed rubric statistics
    attempted_answer = sum(1 for r in all_rubrics if r.num_correct_handles > 0 or r.score > 0)
    found_correct_profile = sum(1 for r in all_rubrics if r.num_correct_handles > 0)
    
    results = {
        "checkpoint": str(checkpoint_path),
        "num_queries": len(queries),
        "num_rollouts_per_query": num_rollouts,
        "accuracy": accuracy,
        "correct_answers": correct_answers,
        "total_samples": len(all_scores),
        "attempted_answer": attempted_answer,
        "avg_reward": avg_reward,
        "median_reward": median_reward,
        "std_reward": std_reward,
        "min_reward": float(min(all_rewards)) if all_rewards else 0.0,
        "max_reward": float(max(all_rewards)) if all_rewards else 0.0,
        "avg_score": avg_score,
        "avg_hits": avg_hits,
        "found_correct_profile": found_correct_profile,
        "avg_turns": avg_turns,
        "eval_time": eval_time,
    }
    
    return results


def print_results(results: Dict):
    """Print evaluation results in a nice format."""
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    print(f"Checkpoint: {results['checkpoint']}")
    print(f"Queries evaluated: {results['num_queries']} x {results['num_rollouts_per_query']} rollout(s)")
    print("")
    print(f"🎯 Accuracy: {results['accuracy']:.2%} ({results['correct_answers']}/{results['total_samples']})")
    print(f"📈 Avg Score: {results['avg_score']:.3f}")
    print(f"🎁 Avg Reward: {results['avg_reward']:.3f} (median: {results['median_reward']:.3f}, std: {results['std_reward']:.3f})")
    print(f"🎲 Reward Range: [{results['min_reward']:.3f}, {results['max_reward']:.3f}]")
    print("")
    print(f"📋 Detailed Stats:")
    print(f"  - Attempted answers: {results['attempted_answer']}/{results['total_samples']} ({results['attempted_answer']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  - Found correct profile: {results['found_correct_profile']}/{results['total_samples']} ({results['found_correct_profile']/max(results['total_samples'], 1)*100:.1f}%)")
    print(f"  - Avg hits per query: {results['avg_hits']:.2f}")
    print(f"  - Avg turns per query: {results['avg_turns']:.2f}")
    print(f"  - Evaluation time: {results['eval_time']:.1f}s ({results['eval_time']/results['num_queries']:.2f}s/query)")
    print("="*80)
    print("")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Link Search Agent model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., outputs/grpo_linksearch_masked/final)"
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=100,
        help="Number of evaluation queries (default: 100)"
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of rollouts per query for robustness testing (default: 1)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum turns per query (default: 15)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens per query (default: 4096)"
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=10,
        help="Maximum profiles to return (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON file"
    )
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print("="*80)
    print("🔍 Link Search Agent Evaluation")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluation size: {args.eval_size}")
    print(f"Rollouts per query: {args.num_rollouts}")
    print(f"Split: {args.split}")
    print("="*80)
    print("")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(str(checkpoint_path))
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load evaluation queries
    logger.info(f"Loading {args.split} queries...")
    eval_queries = load_link_search_queries(
        split=args.split,
        limit=args.eval_size,
        shuffle=True,
        seed=42,
    )
    logger.info(f"✓ Loaded {len(eval_queries)} queries")
    
    # Policy configuration
    policy_config = PolicyConfig(
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        max_profiles=args.max_profiles,
        verbose=args.verbose,
        stupid_simple_reward_fn=False,
    )
    
    # Run evaluation
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        evaluate_model(
            model=model,
            tokenizer=tokenizer,
            queries=eval_queries,
            policy_config=policy_config,
            num_rollouts=args.num_rollouts,
            verbose=args.verbose,
        )
    )
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.save_results:
        checkpoint_name = checkpoint_path.name
        output_file = checkpoint_path.parent / f"eval_results_{checkpoint_name}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {output_file}")
        print("")


if __name__ == "__main__":
    main()
