#!/usr/bin/env python3
"""
Test script to verify the environment setup

Usage:
    python test_setup.py
"""

import sys
import subprocess


def test_import(module_name, package_name=None):
    """Test if a Python module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False


def main():
    print("=" * 80)
    print("Testing Environment Setup for Qwen3-32B GRPO Training")
    print("=" * 80)
    print()
    
    # Test Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Test core packages
    print("Testing core packages:")
    print("-" * 40)
    tests = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("trl", "TRL"),
        ("unsloth", "Unsloth"),
        ("peft", "PEFT"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("bitsandbytes", "BitsAndBytes"),
    ]
    
    results = []
    for module, name in tests:
        results.append(test_import(module, name))
    
    print()
    
    # Test CUDA
    print("Testing CUDA:")
    print("-" * 40)
    cuda_ok = test_cuda()
    print()
    
    # Test optional packages
    print("Testing optional packages:")
    print("-" * 40)
    optional_tests = [
        ("wandb", "Wandb"),
        ("tensorboard", "TensorBoard"),
        ("deepspeed", "DeepSpeed"),
    ]
    
    for module, name in optional_tests:
        test_import(module, name)
    
    print()
    
    # Summary
    print("=" * 80)
    if all(results) and cuda_ok:
        print("✅ All required packages are installed and CUDA is available!")
        print("You're ready to start training.")
    else:
        print("⚠️  Some packages are missing or CUDA is not available.")
        print("Please install missing packages: pip install -r requirements.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
