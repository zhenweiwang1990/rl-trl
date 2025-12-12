"""Link Search Agent for model inference and tool execution."""

import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import asdict

import torch

from link_search_agent.data.types import LinkSearchQuery
from link_search_agent.tools import search_profile, read_profile
from link_search_agent.config import PolicyConfig
from link_search_agent.prompts import create_system_prompt, get_tools_schema
from link_search_agent.rollout import (
    LinkSearchRubric,
    calculate_reward,
    compute_score,
    normalize_handle,
)
from link_search_agent.rollout_logger import RolloutLogBuilder, LinkSearchRolloutLog

logger = logging.getLogger(__name__)


class LinkSearchAgent:
    """Link Search Agent that handles model inference and tool execution.
    
    This agent:
    1. Takes a model and tokenizer
    2. Uses transformers' native tool calling support
    3. Parses tool calls from model output
    4. Executes tools (SQL search, profile reading)
    5. Tracks evaluation metrics
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        policy_config: PolicyConfig,
        rollout_index: int = 0,
        num_rollouts: int = 4,
        enable_detailed_logging: bool = False,
        training_step: int = 0,
    ):
        """Initialize the agent.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            policy_config: Policy configuration
            rollout_index: Index of this rollout within its group
            num_rollouts: Total number of rollouts in the group
            enable_detailed_logging: Whether to accumulate detailed rollout logs
            training_step: Current training step number
        """
        self.model = model
        self.tokenizer = tokenizer
        self.policy_config = policy_config
        self.tools = get_tools_schema()
        self.rollout_index = rollout_index
        self.num_rollouts = num_rollouts
        self.enable_detailed_logging = enable_detailed_logging
        self.training_step = training_step
        self.log_builder: Optional[RolloutLogBuilder] = None
        
        # KV Cache state for multi-turn conversation optimization
        # Stores past_key_values from previous turns to avoid re-computation
        self.past_key_values = None
        self.cached_conversation_length = 0  # Number of messages already processed
        self.cached_input_ids = None  # Cached input IDs for debugging
        
        # Optimize model for inference if available (Unsloth FastLanguageModel)
        # This enables various optimizations like fused attention, etc.
        # Note: We enable this during rollout/eval but preserve training capability
        try:
            from unsloth import FastLanguageModel
            # Check if model supports Unsloth optimization
            if hasattr(model, 'model') or hasattr(model, 'base_model'):
                # For inference, enable optimizations but don't persist state
                # The model should already be optimized if loaded with Unsloth
                pass
        except (ImportError, AttributeError, Exception):
            # Not using Unsloth or optimization failed, continue normally
            pass
    
    async def run_query(
        self,
        query: LinkSearchQuery,
        verbose: bool = False,
    ) -> Tuple[LinkSearchRubric, List[Dict[str, Any]], Optional[LinkSearchRolloutLog]]:
        """Run the agent on a single query.
        
        Args:
            query: The query to process
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of (rubric, conversation_history, rollout_log)
        """
        rubric = LinkSearchRubric()
        rubric.num_gold_handles = len(query.gold_handles)
        
        # Track state
        search_history: List[Dict] = []  # Track SQL queries
        read_history: Set[str] = set()  # Track read handles
        found_handles: Set[str] = set()  # All handles found in searches
        correct_found: Set[str] = set()  # Correct handles found
        gold_set = set(normalize_handle(h) for h in query.gold_handles)
        
        # Final results
        final_results: Dict[str, Any] = {}
        
        # Create initial conversation
        system_prompt = create_system_prompt(
            query, 
            self.policy_config.max_turns,
            self.policy_config.max_profiles,
        )
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Search query: {query.query}"},
        ]
        
        # Calculate temperature
        if self.policy_config.enable_dynamic_temperature:
            temperature = self.policy_config.base_temperature + (
                self.rollout_index * self.policy_config.temperature_increment
            )
            repetition_penalty = self.policy_config.base_repetition_penalty + (
                self.rollout_index * self.policy_config.repetition_penalty_increment
            )
        else:
            temperature = 0.7
            repetition_penalty = 1.0
        
        # Initialize log builder
        if self.enable_detailed_logging:
            self.log_builder = RolloutLogBuilder(
                query=query,
                system_prompt=system_prompt,
                max_turns=self.policy_config.max_turns,
                max_profiles=self.policy_config.max_profiles,
                policy_config={
                    "max_turns": self.policy_config.max_turns,
                    "max_tokens": self.policy_config.max_tokens,
                    "max_profiles": self.policy_config.max_profiles,
                },
                step=self.training_step,
                rollout_index=self.rollout_index,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            for msg in conversation:
                self.log_builder.log_conversation_message(msg)
        
        if verbose:
            print("\n" + "="*80)
            print(f"QUERY {query.id}")
            print("="*80)
            print(f"Search: {query.query}")
            print(f"Gold Handles: {query.gold_handles}")
            print("="*80)
        
        # Agent loop
        for turn in range(self.policy_config.max_turns):
            turn_start_time = time.time()
            rubric.num_turns += 1
            
            # Start turn timing
            if self.log_builder:
                self.log_builder.start_turn(turn + 1)
            
            if verbose:
                print(f"\n{'─'*80}")
                print(f"TURN {turn + 1}/{self.policy_config.max_turns}")
                print(f"{'─'*80}")
            
            try:
                # Generate model response
                llm_start = time.time()
                response_message, raw_content, input_tokens, output_tokens = self._generate_response(
                    conversation, turn, verbose
                )
                llm_time_ms = (time.time() - llm_start) * 1000.0
                
                # Calculate tokens per second for performance monitoring
                tokens_per_sec = (output_tokens / (llm_time_ms / 1000.0)) if llm_time_ms > 0 else 0.0
                
                rubric.total_input_tokens += input_tokens
                rubric.total_output_tokens += output_tokens
                
                # Log performance metrics if verbose
                if verbose and tokens_per_sec > 0:
                    logger.debug(f"Generation speed: {tokens_per_sec:.1f} tokens/s")
                
                # Log LLM generation metrics
                if self.log_builder:
                    self.log_builder.log_llm_generation(
                        generation_time_ms=llm_time_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        raw_output=raw_content,
                    )
                
                if verbose:
                    print(f"⏱️  LLM Generation: {llm_time_ms:.2f}ms | "
                          f"Tokens: {input_tokens} in / {output_tokens} out")
                    
                    # Show raw model output for debugging (controlled by show_raw_output flag)
                    if self.policy_config.show_raw_output and raw_content:
                        print(f"\n🔍 Raw Model Output ({len(raw_content)} chars, {output_tokens} tokens):")
                        print(f"{'─'*80}")
                        print(raw_content[:1000])  # Show first 1000 chars
                        if len(raw_content) > 1000:
                            print(f"\n... (truncated, {len(raw_content) - 1000} more chars)")
                        print(f"{'─'*80}")
                
                # Add assistant message
                assistant_msg = {"role": "assistant"}
                
                # Only add non-None fields to avoid template errors
                content = response_message.get("content")
                tool_calls = response_message.get("tool_calls")
                
                if tool_calls is not None:
                    assistant_msg["tool_calls"] = tool_calls
                    # When tool_calls exist, content should be empty string if None
                    assistant_msg["content"] = content if content is not None else ""
                elif content is not None:
                    assistant_msg["content"] = content
                else:
                    # Fallback: ensure content is at least an empty string
                    assistant_msg["content"] = ""
                
                conversation.append(assistant_msg)
                
                if self.log_builder:
                    self.log_builder.log_conversation_message(assistant_msg)
                
                # Handle tool calls
                if response_message.get("tool_calls"):
                    tools_start = time.time()
                    should_break = await self._execute_tool_calls(
                        response_message["tool_calls"],
                        query,
                        rubric,
                        conversation,
                        search_history,
                        read_history,
                        found_handles,
                        correct_found,
                        gold_set,
                        final_results,
                        verbose,
                        turn_number=turn + 1,
                    )
                    tools_time_ms = (time.time() - tools_start) * 1000.0
                    
                    if verbose and response_message.get("tool_calls"):
                        print(f"⏱️  Tools Execution: {tools_time_ms:.2f}ms")
                    
                    if should_break:
                        break
                
                elif response_message.get("content"):
                    # Model returned text, try to parse as results
                    parsed_results = self._try_parse_results(response_message["content"])
                    if parsed_results:
                        final_results = parsed_results
                        rubric.attempted_answer = True
                        break
                    else:
                        rubric.cant_parse_tool_call = True
                        if verbose:
                            print(f"\n⚠️ Model returned text instead of tool call")
                        break
                else:
                    rubric.cant_parse_tool_call = True
                    if verbose:
                        print(f"\n⚠️ Model returned empty response")
                    break
                    
            except Exception as e:
                logger.error(f"Error in agent loop turn {turn + 1}: {e}")
                if verbose:
                    print(f"\n❌ Exception: {e}")
                    import traceback
                    traceback.print_exc()
                rubric.cant_parse_tool_call = True
                break
            finally:
                # Finish turn timing
                turn_total_time_ms = (time.time() - turn_start_time) * 1000.0
                if self.log_builder:
                    self.log_builder.finish_turn(turn_total_time_ms)
                
                if verbose:
                    print(f"⏱️  Turn Total Time: {turn_total_time_ms:.2f}ms")
        
        # Check if ran out of turns
        if rubric.num_turns >= self.policy_config.max_turns and not rubric.attempted_answer:
            rubric.ran_out_of_turns = True
            if verbose:
                print(f"\n⏱️ Agent ran out of turns")
        
        # Compute final score
        predicted_handles = list(final_results.keys()) if final_results else []
        score_result = compute_score(predicted_handles, query.gold_handles)
        
        rubric.num_predicted_handles = len(predicted_handles)
        rubric.num_correct_handles = score_result["num_hits"]
        rubric.score = score_result["score"]
        rubric.precision = score_result["precision"]
        rubric.recall = score_result["recall"]
        
        if verbose:
            self._print_evaluation_summary(rubric, query, predicted_handles, score_result)
        
        # Build rollout log
        rollout_log = None
        if self.log_builder:
            reward = calculate_reward(self.policy_config, rubric)
            self.log_builder.log_final_results(final_results, predicted_handles)
            rollout_log = self.log_builder.build(rubric, reward)
        
        return rubric, conversation, rollout_log
    
    def _generate_response(
        self,
        conversation: List[Dict],
        turn: int,
        verbose: bool,
    ) -> Tuple[Dict[str, Any], str, int, int]:
        """Generate a response from the model with KV cache optimization.
        
        This implementation uses incremental tokenization and attempts to reuse
        KV cache across turns. For the first turn, we tokenize the full conversation.
        For subsequent turns, we tokenize incrementally and try to reuse cached KV values.
        
        Args:
            conversation: Full conversation history
            turn: Current turn number (0-indexed)
            verbose: Whether to print debug info
            
        Returns:
            Tuple of (response_message, raw_content, input_tokens, output_tokens)
        """
        device = self.model.device
        is_first_turn = (turn == 0)
        
        # Calculate temperature
        if self.policy_config.enable_dynamic_temperature:
            temperature = self.policy_config.base_temperature + (
                self.rollout_index * self.policy_config.temperature_increment
            )
            repetition_penalty = self.policy_config.base_repetition_penalty + (
                self.rollout_index * self.policy_config.repetition_penalty_increment
            )
        else:
            temperature = 0.7
            repetition_penalty = 1.0
        
        # Optimize max_new_tokens
        max_new_tokens = min(self.policy_config.max_tokens, 512)
        
        # Tokenization: Use full conversation (transformers handles KV cache internally)
        # The real optimization comes from avoiding redundant tokenization work
        # and letting the model's internal KV cache work during generation
        try:
            text = self.tokenizer.apply_chat_template(
                conversation,
                tools=self.tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        input_tokens = input_ids.shape[1]
        
        # Track tokenization savings
        if not is_first_turn and verbose:
            logger.debug(
                f"Turn {turn + 1}: Tokenization ({input_tokens} tokens, "
                f"conversation: {len(conversation)} messages)"
            )
        
        # Use inference mode for faster generation
        with torch.inference_mode():
            was_training = self.model.training
            if was_training:
                self.model.eval()
            
            try:
                # For KV cache reuse, we'll use a strategy where we process the context
                # through forward() first if we have cached state, then generate
                if not is_first_turn and self.past_key_values is not None and self.cached_input_ids is not None:
                    # Try to use cached KV values
                    cached_length = self.cached_input_ids.shape[1]
                    
                    if input_ids.shape[1] > cached_length:
                        # Process only the new tokens through forward()
                        new_input_ids = input_ids[:, cached_length:]
                        
                        # Forward pass to update past_key_values
                        model_outputs = self.model(
                            input_ids=new_input_ids,
                            past_key_values=self.past_key_values,
                            use_cache=True,
                        )
                        
                        # Update past_key_values
                        self.past_key_values = model_outputs.past_key_values
                        
                        # Now we need to generate from the updated hidden states
                        # Since generate() doesn't easily accept past_key_values in this format,
                        # we'll use the logits to manually generate or fall back to standard generate
                        # For now, use standard generate but the forward pass already updated the cache
                        # This is a limitation - we can't easily pass past_key_values to generate()
                        pass
                
                # Standard generation (KV cache is handled internally by transformers)
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,  # Enable KV cache (works within single generate() call)
                    num_beams=1,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                
                # Note: We can't easily extract past_key_values from generate() without
                # using return_dict_in_generate=True, which would require significant refactoring.
                # For now, we cache the input_ids length to track progress.
                # The model's internal KV cache optimization still helps within each generate() call.
                
            finally:
                if was_training:
                    self.model.train()
        
        # Update cached state
        # Cache input_ids (context) before generation for reference
        self.cached_input_ids = input_ids
        self.cached_conversation_length = len(conversation)
        
        output_tokens = outputs.shape[1] - input_ids.shape[1]
        
        # Decode only the generated part
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        response_message, _ = self._parse_tool_calls_from_response(response, verbose)
        
        if verbose:
            print(f"\n📝 Model Response:")
            if response_message.get("tool_calls"):
                print(f"   Tool Calls: {len(response_message['tool_calls'])}")
                for tc in response_message["tool_calls"]:
                    func_name = tc['function']['name']
                    func_args = tc['function']['arguments'][:100]
                    print(f"   - {func_name}: {func_args}")
            elif response_message.get("content"):
                content = response_message["content"]
                print(f"   {content[:300]}...")
        
        return response_message, response, input_tokens, output_tokens
    
    def _parse_tool_calls_from_response(
        self,
        response: str,
        verbose: bool,
    ) -> Tuple[Dict[str, Any], bool]:
        """Parse tool calls from model response."""
        # Try <tool_call> tags
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        if tool_call_matches:
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                try:
                    tool_data = json.loads(match.strip())
                    tool_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tool_data.get("name", ""),
                            "arguments": json.dumps(tool_data.get("arguments", {})),
                        }
                    })
                except json.JSONDecodeError:
                    continue
            
            if tool_calls:
                return {"content": None, "tool_calls": tool_calls}, True
        
        # Try direct JSON tool call
        try:
            json_match = re.search(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{(?:[^{}]|{[^{}]*})*\}', response, re.DOTALL)
            
            if json_match:
                parsed = json.loads(json_match.group())
                if "name" in parsed and "arguments" in parsed:
                    tool_calls = [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed["arguments"]),
                        }
                    }]
                    return {"content": None, "tool_calls": tool_calls}, True
        except (json.JSONDecodeError, KeyError):
            pass
        
        return {"content": response, "tool_calls": None}, True
    
    def _try_parse_results(self, content: str) -> Optional[Dict[str, Any]]:
        """Try to parse final results from text content."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*"results"[\s\S]*\}', content)
            if json_match:
                parsed = json.loads(json_match.group())
                if "results" in parsed:
                    return parsed["results"]
            
            # Try direct parse
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                if "results" in parsed:
                    return parsed["results"]
                return parsed
        except:
            pass
        return None
    
    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        query: LinkSearchQuery,
        rubric: LinkSearchRubric,
        conversation: List[Dict],
        search_history: List[Dict],
        read_history: Set[str],
        found_handles: Set[str],
        correct_found: Set[str],
        gold_set: Set[str],
        final_results: Dict[str, Any],
        verbose: bool,
        turn_number: int = 0,
    ) -> bool:
        """Execute tool calls."""
        should_break = False
        
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", "")
            tool_function = tool_call.get("function", {})
            tool_name = tool_function.get("name")
            
            arguments_str = tool_function.get("arguments", "{}")
            try:
                tool_args = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except json.JSONDecodeError as e:
                rubric.bad_tool_call_args = True
                if verbose:
                    print(f"\n❌ Failed to parse tool arguments: {e}")
                should_break = True
                break
            
            if not tool_name:
                rubric.bad_tool_call_name = True
                should_break = True
                break
            
            if verbose:
                print(f"\n🔧 Tool Call: {tool_name}")
                print(f"   Arguments: {json.dumps(tool_args, indent=2)[:200]}")
            
            # Execute tool
            tool_exec_start = time.time()
            tool_result, should_break_inner = await self._execute_single_tool(
                tool_name,
                tool_args,
                query,
                rubric,
                search_history,
                read_history,
                found_handles,
                correct_found,
                gold_set,
                final_results,
                verbose,
            )
            tool_exec_time_ms = (time.time() - tool_exec_start) * 1000.0
            
            if verbose:
                print(f"\n📊 Tool Result:")
                result_str = json.dumps(tool_result, indent=2)
                print(f"   {result_str[:300]}...")
                print(f"⏱️  Tool Execution Time: {tool_exec_time_ms:.2f}ms")
            
            # Add tool result to conversation
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result),
            }
            conversation.append(tool_msg)
            
            if self.log_builder:
                self.log_builder.log_conversation_message(tool_msg)
                self.log_builder.log_tool_call(
                    turn_number=turn_number,
                    tool_name=tool_name,
                    tool_arguments=tool_args,
                    tool_result=tool_result,
                    error=tool_result.get("error") if isinstance(tool_result, dict) else None,
                    execution_time_ms=tool_exec_time_ms,
                )
            
            if should_break_inner:
                should_break = True
                break
        
        return should_break
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        query: LinkSearchQuery,
        rubric: LinkSearchRubric,
        search_history: List[Dict],
        read_history: Set[str],
        found_handles: Set[str],
        correct_found: Set[str],
        gold_set: Set[str],
        final_results: Dict[str, Any],
        verbose: bool,
    ) -> Tuple[Any, bool]:
        """Execute a single tool call."""
        should_break = False
        
        if tool_name == "search_profile":
            rubric.num_total_searches += 1
            
            sql = tool_args.get("sql", "")
            params = tool_args.get("params")
            max_rows = tool_args.get("max_rows", 200)
            
            # Check for repeated search
            sql_normalized = sql.strip().lower()
            is_repeat = any(h["sql"].strip().lower() == sql_normalized for h in search_history)
            
            if is_repeat:
                rubric.num_repeated_searches += 1
                if verbose:
                    print(f"\n⚠️ Repeated search query!")
            else:
                rubric.num_unique_searches += 1
            
            # Execute search
            result = search_profile(sql, params, max_rows)
            
            if not result["success"]:
                rubric.sql_error_count += 1
                if verbose:
                    print(f"\n❌ SQL Error: {result.get('error')}")
            else:
                result_count = result["rowCount"]
                
                if result_count == 0:
                    rubric.num_zero_result_searches += 1
                else:
                    rubric.num_searches_with_results += 1
                
                # Track handles found
                for row in result.get("rows", []):
                    handle = row.get("handle") or row.get("linkedin_handle")
                    if handle:
                        handle_norm = normalize_handle(handle)
                        found_handles.add(handle_norm)
                        if handle_norm in gold_set:
                            if handle_norm not in correct_found:
                                correct_found.add(handle_norm)
                                if rubric.turn_first_correct_found < 0:
                                    rubric.turn_first_correct_found = rubric.num_turns
                
                # Analyze search strategy
                if len(search_history) > 0:
                    prev = search_history[-1]
                    if prev["count"] == 0 and result_count > 0:
                        rubric.broadened_search_after_zero += 1
                    elif prev["count"] >= 10 and result_count < 10:
                        rubric.narrowed_search_after_many += 1
                
                if verbose:
                    correct_in_results = [h for h in gold_set if any(
                        normalize_handle(r.get("handle") or r.get("linkedin_handle", "")) == h
                        for r in result.get("rows", [])
                    )]
                    print(f"\n✓ Search returned {result_count} results")
                    if correct_in_results:
                        print(f"   ✓ CORRECT handles found: {correct_in_results}")
            
            search_history.append({
                "sql": sql,
                "count": result.get("rowCount", 0),
                "turn": rubric.num_turns,
            })
            
            return result, should_break
        
        elif tool_name == "read_profile":
            rubric.num_total_reads += 1
            
            handle = tool_args.get("linkedin_handle", "").strip().lower()
            
            if handle in read_history:
                rubric.num_repeated_reads += 1
                if verbose:
                    print(f"\n⚠️ Repeated read of {handle}")
            else:
                rubric.num_unique_reads += 1
                read_history.add(handle)
                
                # Check if reading after good search
                if len(search_history) > 0:
                    last_search = search_history[-1]
                    if last_search["turn"] == rubric.num_turns - 1 and 0 < last_search["count"] < 20:
                        rubric.read_after_good_search += 1
            
            result = read_profile(handle)
            
            if not result["success"]:
                rubric.num_invalid_reads += 1
                if verbose:
                    print(f"\n❌ Read failed: {result.get('error')}")
            else:
                if handle in gold_set:
                    rubric.num_correct_profiles_read += 1
                    if rubric.turn_first_correct_read < 0:
                        rubric.turn_first_correct_read = rubric.num_turns
                    if verbose:
                        print(f"\n✓ Read CORRECT profile: {handle}")
            
            return result, should_break
        
        elif tool_name == "return_results":
            rubric.attempted_answer = True
            
            results = tool_args.get("results", {})
            if isinstance(results, dict):
                final_results.update(results)
            
            if verbose:
                print(f"\n🎯 Agent returning {len(results)} results")
            
            return {"status": "Results submitted", "count": len(results)}, True
        
        else:
            rubric.bad_tool_call_name = True
            if verbose:
                print(f"\n❌ Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}, True
    
    def _print_evaluation_summary(
        self,
        rubric: LinkSearchRubric,
        query: LinkSearchQuery,
        predicted_handles: List[str],
        score_result: Dict,
    ):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Turns used: {rubric.num_turns}/{self.policy_config.max_turns}")
        print(f"Predicted handles: {predicted_handles}")
        print(f"Gold handles: {query.gold_handles}")
        print(f"\n📊 Score: {rubric.score:.3f}")
        print(f"   Hits: {score_result['num_hits']}/{score_result['num_gold']}")
        print(f"   Precision: {rubric.precision:.3f}")
        print(f"   Recall: {rubric.recall:.3f}")
        
        print(f"\n🔍 Search Metrics:")
        print(f"   Unique searches: {rubric.num_unique_searches}")
        print(f"   Repeated searches: {rubric.num_repeated_searches}")
        print(f"   Zero-result searches: {rubric.num_zero_result_searches}")
        
        print(f"\n📖 Read Metrics:")
        print(f"   Unique reads: {rubric.num_unique_reads}")
        print(f"   Correct profiles read: {rubric.num_correct_profiles_read}")
        
        reward = calculate_reward(self.policy_config, rubric)
        print(f"\n🎯 Final Reward: {reward:.3f}")
        print(f"{'='*80}\n")

