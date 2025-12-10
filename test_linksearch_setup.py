#!/usr/bin/env python3
"""Test script to verify Link Search Agent setup."""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test link_search_agent imports
        from link_search_agent import LinkSearchAgent, GRPOConfig, PolicyConfig
        from link_search_agent.data import LinkSearchQuery, load_link_search_queries
        from link_search_agent.tools import search_profile, read_profile
        from link_search_agent.prompts import create_system_prompt, get_tools_schema
        from link_search_agent.rollout import LinkSearchRubric, calculate_reward
        from link_search_agent.grpo_utils import execute_rollout, prepare_dataset
        from link_search_agent.trainer import LinkSearchGRPOTrainer
        from link_search_agent.rollout_logger import RolloutLogBuilder, LinkSearchRolloutLog
        
        print("✓ link_search_agent imports successful")
    except Exception as e:
        print(f"✗ link_search_agent imports failed: {e}")
        return False
    
    try:
        # Test grpo imports
        from grpo import AccuracyStopCallback
        from grpo.utils import get_env_int, get_env_float, find_latest_checkpoint
        
        print("✓ grpo imports successful")
    except Exception as e:
        print(f"✗ grpo imports failed: {e}")
        return False
    
    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from link_search_agent import GRPOConfig, PolicyConfig
        
        # Test default config
        grpo_config = GRPOConfig()
        policy_config = PolicyConfig()
        
        print(f"✓ GRPO Config: model={grpo_config.model_name}, rollouts={grpo_config.num_generations}")
        print(f"✓ Policy Config: max_turns={policy_config.max_turns}, max_profiles={policy_config.max_profiles}")
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    return True


def test_tools():
    """Test that tools can be loaded."""
    print("\nTesting tools...")
    
    try:
        from link_search_agent.prompts import get_tools_schema
        
        tools = get_tools_schema()
        tool_names = [t['function']['name'] for t in tools]
        
        print(f"✓ Found {len(tools)} tools: {', '.join(tool_names)}")
        
        expected_tools = {"search_profile", "read_profile", "return_results"}
        if set(tool_names) == expected_tools:
            print("✓ All expected tools present")
        else:
            print(f"✗ Missing tools: {expected_tools - set(tool_names)}")
            return False
    except Exception as e:
        print(f"✗ Tools test failed: {e}")
        return False
    
    return True


def test_reward_function():
    """Test reward function calculation."""
    print("\nTesting reward function...")
    
    try:
        from link_search_agent import PolicyConfig
        from link_search_agent.rollout import LinkSearchRubric, calculate_reward
        
        policy_config = PolicyConfig()
        
        # Test perfect case
        rubric = LinkSearchRubric(
            score=1.0,
            num_correct_handles=10,
            num_gold_handles=10,
            num_predicted_handles=10,
            attempted_answer=True,
            num_turns=5,
        )
        
        reward = calculate_reward(policy_config, rubric)
        print(f"✓ Perfect case reward: {reward:.2f}")
        
        # Test failure case
        rubric_fail = LinkSearchRubric(
            score=0.0,
            num_correct_handles=0,
            num_gold_handles=10,
            num_predicted_handles=5,
            attempted_answer=True,
            num_turns=15,
        )
        
        reward_fail = calculate_reward(policy_config, rubric_fail)
        print(f"✓ Failure case reward: {reward_fail:.2f}")
        
    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Link Search Agent Setup Test")
    print("="*60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_config()
    all_passed &= test_tools()
    all_passed &= test_reward_function()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed!")
        print("\nYou can now run training with:")
        print("  python train_grpo_linksearch.py --mode masked")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
