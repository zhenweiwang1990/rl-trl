"""Detailed rollout logging for Link Search Agent debugging and analysis."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from link_search_agent.data.types import LinkSearchQuery
from link_search_agent.rollout import LinkSearchRubric

logger = logging.getLogger(__name__)


@dataclass
class ToolCallLog:
    """Log entry for a single tool call."""
    turn_number: int
    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any
    handles_found: List[str] = None  # Handles found in search results
    correct_handles_found: List[str] = None  # Which of those are correct
    is_correct_profile_read: bool = False  # For read_profile
    result_count: int = 0  # Number of results
    error: Optional[str] = None
    timestamp: str = ""
    execution_time_ms: float = 0.0  # Tool execution time in milliseconds
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.handles_found is None:
            self.handles_found = []
        if self.correct_handles_found is None:
            self.correct_handles_found = []


@dataclass
class TurnTimingLog:
    """Detailed timing information for a single turn."""
    turn_number: int
    llm_generation_time_ms: float = 0.0  # LLM generation time
    llm_input_tokens: int = 0  # Input tokens count
    llm_output_tokens: int = 0  # Output tokens count
    llm_raw_output: str = ""  # Raw model output (for debugging thinking)
    llm_raw_output_length: int = 0  # Length of raw output
    tool_execution_time_ms: float = 0.0  # Total tool execution time for this turn
    turn_total_time_ms: float = 0.0  # Total time for this turn
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class LinkSearchRolloutLog:
    """Complete log for a single link search rollout."""
    # Query information
    query_id: str
    query_text: str
    gold_handles: List[str]
    
    # System configuration
    system_prompt: str
    max_turns: int
    max_profiles: int
    policy_config: Dict[str, Any]
    
    # Rollout metadata
    step: int
    rollout_index: int
    temperature: float
    repetition_penalty: float
    
    # Execution trace
    conversation_history: List[Dict[str, Any]]
    tool_calls: List[ToolCallLog]
    turn_timings: List[TurnTimingLog] = None  # Detailed timing for each turn
    
    # Results
    predicted_handles: List[str] = None
    final_results: Dict[str, Any] = None
    
    # Evaluation
    rubric: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    
    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    query_total_time_ms: float = 0.0  # Total query processing time in ms
    
    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now().isoformat()
        if self.predicted_handles is None:
            self.predicted_handles = []
        if self.tool_calls is None:
            self.tool_calls = []
        if self.turn_timings is None:
            self.turn_timings = []


class RolloutLogBuilder:
    """Builder for constructing LinkSearchRolloutLog during rollout execution."""
    
    def __init__(
        self,
        query: LinkSearchQuery,
        system_prompt: str,
        max_turns: int,
        max_profiles: int,
        policy_config: Dict[str, Any],
        step: int,
        rollout_index: int,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
    ):
        """Initialize the log builder."""
        import time
        self.start_time = time.time()
        
        self.log = LinkSearchRolloutLog(
            query_id=query.id,
            query_text=query.query,
            gold_handles=query.gold_handles,
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_profiles=max_profiles,
            policy_config=policy_config,
            step=step,
            rollout_index=rollout_index,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            conversation_history=[],
            tool_calls=[],
            turn_timings=[],
            start_time=datetime.now().isoformat(),
        )
        
        self.gold_handles_set = set(h.lower() for h in query.gold_handles)
        self.current_turn_timing: Optional[TurnTimingLog] = None
    
    def log_conversation_message(self, message: Dict[str, Any]) -> None:
        """Log a conversation message."""
        clean_message = {
            "role": message.get("role"),
            "content": message.get("content"),
        }
        
        if message.get("tool_calls"):
            clean_message["tool_calls"] = message["tool_calls"]
        
        if message.get("tool_call_id"):
            clean_message["tool_call_id"] = message["tool_call_id"]
        
        self.log.conversation_history.append(clean_message)
    
    def start_turn(self, turn_number: int) -> None:
        """Start timing a new turn."""
        self.current_turn_timing = TurnTimingLog(turn_number=turn_number)
    
    def log_llm_generation(
        self,
        generation_time_ms: float,
        input_tokens: int,
        output_tokens: int,
        raw_output: str = "",
    ) -> None:
        """Log LLM generation metrics."""
        if self.current_turn_timing:
            self.current_turn_timing.llm_generation_time_ms = generation_time_ms
            self.current_turn_timing.llm_input_tokens = input_tokens
            self.current_turn_timing.llm_output_tokens = output_tokens
            self.current_turn_timing.llm_raw_output = raw_output
            self.current_turn_timing.llm_raw_output_length = len(raw_output)
    
    def log_tool_call(
        self,
        turn_number: int,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        tool_result: Any,
        error: Optional[str] = None,
        execution_time_ms: float = 0.0,
    ) -> None:
        """Log a tool call execution."""
        handles_found = []
        correct_handles_found = []
        is_correct_profile_read = False
        result_count = 0
        
        if tool_name == "search_profile":
            # Extract handles from search results
            if isinstance(tool_result, dict) and tool_result.get("success"):
                rows = tool_result.get("rows", [])
                result_count = len(rows)
                for row in rows:
                    handle = row.get("handle") or row.get("linkedin_handle")
                    if handle:
                        handle_lower = handle.lower()
                        handles_found.append(handle_lower)
                        if handle_lower in self.gold_handles_set:
                            correct_handles_found.append(handle_lower)
        
        elif tool_name == "read_profile":
            result_count = 1 if (isinstance(tool_result, dict) and tool_result.get("success")) else 0
            handle = tool_arguments.get("linkedin_handle", "").lower()
            if handle in self.gold_handles_set:
                is_correct_profile_read = True
        
        elif tool_name == "return_results":
            # Extract predicted handles from results
            results = tool_arguments.get("results", {})
            if isinstance(results, dict):
                handles_found = list(results.keys())
                for h in handles_found:
                    if h.lower() in self.gold_handles_set:
                        correct_handles_found.append(h.lower())
        
        tool_log = ToolCallLog(
            turn_number=turn_number,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            handles_found=handles_found,
            correct_handles_found=correct_handles_found,
            is_correct_profile_read=is_correct_profile_read,
            result_count=result_count,
            error=error,
            execution_time_ms=execution_time_ms,
        )
        
        self.log.tool_calls.append(tool_log)
        
        # Accumulate tool execution time for current turn
        if self.current_turn_timing:
            self.current_turn_timing.tool_execution_time_ms += execution_time_ms
    
    def finish_turn(self, turn_total_time_ms: float) -> None:
        """Finish timing the current turn."""
        if self.current_turn_timing:
            self.current_turn_timing.turn_total_time_ms = turn_total_time_ms
            self.log.turn_timings.append(self.current_turn_timing)
            self.current_turn_timing = None
    
    def log_final_results(
        self,
        results: Dict[str, Any],
        predicted_handles: List[str],
    ) -> None:
        """Log the final results."""
        self.log.final_results = results
        self.log.predicted_handles = predicted_handles
    
    def build(
        self,
        rubric: LinkSearchRubric,
        reward: float,
    ) -> LinkSearchRolloutLog:
        """Finalize and return the completed log."""
        import time
        
        self.log.duration_seconds = time.time() - self.start_time
        self.log.query_total_time_ms = self.log.duration_seconds * 1000.0
        self.log.end_time = datetime.now().isoformat()
        self.log.rubric = rubric.to_metrics()
        self.log.reward = reward
        self.log.total_input_tokens = rubric.total_input_tokens
        self.log.total_output_tokens = rubric.total_output_tokens
        
        return self.log


def save_rollout_logs(
    rollout_logs: List[LinkSearchRolloutLog],
    output_dir: str = "outputs/rollout_logs",
) -> List[str]:
    """Save multiple rollout logs to disk.
    
    Args:
        rollout_logs: List of rollout logs to save
        output_dir: Base directory for logs
        
    Returns:
        List of file paths where logs were saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for rollout_log in rollout_logs:
        # Create directory structure: step_X/query_Y/
        step_dir = output_path / f"step_{rollout_log.step}"
        query_dir = step_dir / f"query_{rollout_log.query_id}"
        query_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as rollout_Z.json
        filename = f"rollout_{rollout_log.rollout_index}.json"
        filepath = query_dir / filename
        
        # Convert to dict
        log_dict = asdict(rollout_log)
        
        # Convert nested dataclasses
        log_dict["tool_calls"] = [asdict(tc) for tc in rollout_log.tool_calls]
        log_dict["turn_timings"] = [asdict(tt) for tt in rollout_log.turn_timings]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False)
        
        saved_paths.append(str(filepath))
    
    if saved_paths:
        logger.info(f"Saved {len(saved_paths)} rollout logs to: {output_dir}")
        
        # Log timing summary
        _log_timing_summary(rollout_logs)
    
    return saved_paths


def _log_timing_summary(rollout_logs: List[LinkSearchRolloutLog]) -> None:
    """Log detailed timing summary for rollout logs."""
    if not rollout_logs:
        return
    
    # Aggregate timing statistics
    query_times = [log.query_total_time_ms for log in rollout_logs]
    all_turn_times = []
    all_llm_times = []
    all_tool_times = []
    all_input_tokens = []
    all_output_tokens = []
    
    for log in rollout_logs:
        for turn_timing in log.turn_timings:
            all_turn_times.append(turn_timing.turn_total_time_ms)
            all_llm_times.append(turn_timing.llm_generation_time_ms)
            all_tool_times.append(turn_timing.tool_execution_time_ms)
            all_input_tokens.append(turn_timing.llm_input_tokens)
            all_output_tokens.append(turn_timing.llm_output_tokens)
    
    # Calculate statistics
    import numpy as np
    
    logger.info("="*60)
    logger.info("ROLLOUT TIMING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total rollouts logged: {len(rollout_logs)}")
    logger.info(f"Total turns: {len(all_turn_times)}")
    logger.info("")
    logger.info(f"Query-level timing:")
    logger.info(f"  Avg: {np.mean(query_times):.2f}ms")
    logger.info(f"  Min: {np.min(query_times):.2f}ms")
    logger.info(f"  Max: {np.max(query_times):.2f}ms")
    logger.info(f"  Std: {np.std(query_times):.2f}ms")
    logger.info("")
    
    if all_turn_times:
        logger.info(f"Turn-level timing:")
        logger.info(f"  Avg turn total: {np.mean(all_turn_times):.2f}ms")
        logger.info(f"  Avg LLM generation: {np.mean(all_llm_times):.2f}ms")
        logger.info(f"  Avg tool execution: {np.mean(all_tool_times):.2f}ms")
        logger.info("")
        logger.info(f"Token statistics:")
        logger.info(f"  Avg input tokens per turn: {np.mean(all_input_tokens):.0f}")
        logger.info(f"  Avg output tokens per turn: {np.mean(all_output_tokens):.0f}")
        logger.info(f"  Total input tokens: {sum(all_input_tokens)}")
        logger.info(f"  Total output tokens: {sum(all_output_tokens)}")
    
    logger.info("="*60)

