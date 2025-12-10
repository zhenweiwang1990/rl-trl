#!/usr/bin/env python3
"""Analyze rollout logs to identify performance issues and thinking content."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse


def analyze_single_log(log_path: Path) -> Dict[str, Any]:
    """Analyze a single rollout log file."""
    with open(log_path, 'r') as f:
        log = json.load(f)
    
    analysis = {
        "file": str(log_path),
        "query_id": log.get("query_id"),
        "step": log.get("step"),
        "rollout_index": log.get("rollout_index"),
        "query_total_time_ms": log.get("query_total_time_ms", 0),
        "total_turns": len(log.get("turn_timings", [])),
        "total_input_tokens": log.get("total_input_tokens", 0),
        "total_output_tokens": log.get("total_output_tokens", 0),
        "reward": log.get("reward", 0),
        "score": log.get("rubric", {}).get("score", 0),
    }
    
    # Analyze turn timings
    turn_timings = log.get("turn_timings", [])
    if turn_timings:
        analysis["turns"] = []
        for turn in turn_timings:
            turn_analysis = {
                "turn_number": turn.get("turn_number"),
                "llm_time_ms": turn.get("llm_generation_time_ms", 0),
                "tool_time_ms": turn.get("tool_execution_time_ms", 0),
                "total_time_ms": turn.get("turn_total_time_ms", 0),
                "input_tokens": turn.get("llm_input_tokens", 0),
                "output_tokens": turn.get("llm_output_tokens", 0),
                "raw_output_length": turn.get("llm_raw_output_length", 0),
                "tokens_per_second": (
                    turn.get("llm_output_tokens", 0) / (turn.get("llm_generation_time_ms", 1) / 1000.0)
                    if turn.get("llm_generation_time_ms", 0) > 0 else 0
                ),
            }
            
            # Check if there's thinking content (raw output much longer than tool call)
            raw_output = turn.get("llm_raw_output", "")
            if raw_output:
                turn_analysis["has_raw_output"] = True
                turn_analysis["raw_output_preview"] = raw_output[:200]
                
                # Heuristic: if raw output is much longer than expected for a tool call,
                # it probably contains thinking
                expected_tool_call_length = 200  # Rough estimate for a tool call
                if len(raw_output) > expected_tool_call_length * 2:
                    turn_analysis["likely_has_thinking"] = True
                    turn_analysis["thinking_ratio"] = len(raw_output) / expected_tool_call_length
            
            analysis["turns"].append(turn_analysis)
        
        # Summary statistics
        llm_times = [t.get("llm_generation_time_ms", 0) for t in turn_timings]
        output_tokens = [t.get("llm_output_tokens", 0) for t in turn_timings]
        
        analysis["avg_llm_time_ms"] = sum(llm_times) / len(llm_times) if llm_times else 0
        analysis["max_llm_time_ms"] = max(llm_times) if llm_times else 0
        analysis["avg_output_tokens"] = sum(output_tokens) / len(output_tokens) if output_tokens else 0
        analysis["max_output_tokens"] = max(output_tokens) if output_tokens else 0
        
        # Calculate overall tokens/second
        total_llm_time_s = sum(llm_times) / 1000.0
        total_output = sum(output_tokens)
        analysis["overall_tokens_per_second"] = (
            total_output / total_llm_time_s if total_llm_time_s > 0 else 0
        )
    
    return analysis


def find_rollout_logs(base_dir: Path) -> List[Path]:
    """Find all rollout log files."""
    return sorted(base_dir.glob("step_*/query_*/rollout_*.json"))


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout timing logs")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="outputs/rollout_logs",
        help="Directory containing rollout logs"
    )
    parser.add_argument(
        "--show-raw-output",
        action="store_true",
        help="Show raw model output for turns with thinking"
    )
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=500,
        help="Only show turns with output tokens >= this threshold"
    )
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        print(f"   Make sure to run training with --enable-detailed-logging flag")
        sys.exit(1)
    
    log_files = find_rollout_logs(logs_dir)
    if not log_files:
        print(f"❌ No rollout logs found in {logs_dir}")
        sys.exit(1)
    
    print(f"📊 Found {len(log_files)} rollout logs")
    print("="*80)
    
    # Analyze all logs
    all_analyses = []
    for log_file in log_files:
        try:
            analysis = analyze_single_log(log_file)
            all_analyses.append(analysis)
        except Exception as e:
            print(f"⚠️  Error analyzing {log_file}: {e}")
    
    # Print summary
    print("\n📈 Performance Summary")
    print("="*80)
    
    if all_analyses:
        avg_tokens_per_sec = sum(a.get("overall_tokens_per_second", 0) for a in all_analyses) / len(all_analyses)
        avg_output_tokens = sum(a.get("avg_output_tokens", 0) for a in all_analyses) / len(all_analyses)
        max_output_tokens = max(a.get("max_output_tokens", 0) for a in all_analyses)
        
        print(f"Average tokens/second: {avg_tokens_per_sec:.2f}")
        print(f"Average output tokens per turn: {avg_output_tokens:.1f}")
        print(f"Max output tokens in any turn: {max_output_tokens}")
    
    # Find and display turns with excessive output
    print(f"\n🔍 Turns with >= {args.min_output_tokens} output tokens:")
    print("="*80)
    
    found_excessive = False
    for analysis in all_analyses:
        for turn in analysis.get("turns", []):
            if turn["output_tokens"] >= args.min_output_tokens:
                found_excessive = True
                print(f"\n📍 Step {analysis['step']}, Query {analysis['query_id']}, "
                      f"Rollout {analysis['rollout_index']}, Turn {turn['turn_number']}")
                print(f"   Output tokens: {turn['output_tokens']}")
                print(f"   LLM time: {turn['llm_time_ms']:.2f}ms")
                print(f"   Speed: {turn['tokens_per_second']:.2f} tokens/s")
                print(f"   Raw output length: {turn.get('raw_output_length', 0)} chars")
                
                if turn.get("likely_has_thinking"):
                    print(f"   ⚠️  Likely contains thinking (ratio: {turn['thinking_ratio']:.1f}x)")
                
                if args.show_raw_output and turn.get("has_raw_output"):
                    print(f"\n   Raw Output Preview:")
                    print(f"   {'-'*76}")
                    preview = turn.get("raw_output_preview", "")
                    for line in preview.split('\n'):
                        print(f"   {line}")
                    print(f"   {'-'*76}")
    
    if not found_excessive:
        print(f"✓ No turns found with >= {args.min_output_tokens} output tokens")
    
    print("\n" + "="*80)
    print(f"📁 Full logs available in: {logs_dir}")


if __name__ == "__main__":
    main()
