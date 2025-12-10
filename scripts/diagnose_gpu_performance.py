#!/usr/bin/env python3
"""Diagnose GPU performance issues for training."""

import sys
import subprocess

def run_cmd(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_cuda():
    """Check CUDA availability and version."""
    print("="*80)
    print("🔍 CUDA Configuration")
    print("="*80)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
                
                # Check memory usage
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"  Memory allocated: {allocated:.2f} GB")
                print(f"  Memory reserved: {reserved:.2f} GB")
        else:
            print("❌ CUDA is not available!")
            print("   This explains why inference is slow!")
    except ImportError:
        print("❌ PyTorch not installed")

def check_flash_attention():
    """Check Flash Attention installation."""
    print("\n" + "="*80)
    print("⚡ Flash Attention Status")
    print("="*80)
    
    # Check flash-attn package
    try:
        import flash_attn
        print(f"✓ flash-attn installed: version {flash_attn.__version__}")
        
        # Try to import key functions
        try:
            from flash_attn import flash_attn_func
            print("✓ flash_attn_func available")
        except ImportError as e:
            print(f"❌ flash_attn_func not available: {e}")
            
    except ImportError:
        print("❌ flash-attn NOT installed")
        print("\nTo install:")
        print("  pip install flash-attn --no-build-isolation")
        print("  or")
        print("  pip install flash-attn==2.5.8 --no-build-isolation")

def check_transformers():
    """Check transformers and dependencies."""
    print("\n" + "="*80)
    print("📦 Dependencies")
    print("="*80)
    
    packages = [
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "trl",
        "unsloth",
    ]
    
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {pkg}: {version}")
        except ImportError:
            print(f"❌ {pkg}: not installed")

def check_gpu_utilization():
    """Check GPU utilization using nvidia-smi."""
    print("\n" + "="*80)
    print("📊 GPU Utilization (nvidia-smi)")
    print("="*80)
    
    output = run_cmd("nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv")
    print(output)
    
    print("\n" + "="*80)
    print("🔥 GPU Power and Temperature")
    print("="*80)
    
    output = run_cmd("nvidia-smi --query-gpu=index,power.draw,power.limit,temperature.gpu --format=csv")
    print(output)

def test_simple_inference():
    """Test simple inference to check performance."""
    print("\n" + "="*80)
    print("🧪 Quick Inference Test")
    print("="*80)
    
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available, skipping test")
            return
        
        device = torch.device("cuda")
        
        # Simple matmul test
        print("\nTesting matrix multiplication...")
        size = 4096
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(3):
            z = torch.matmul(x, y)
        
        torch.cuda.synchronize()
        
        # Timed runs
        times = []
        for _ in range(10):
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        tflops = (2 * size**3) / (avg_time * 1e12)
        
        print(f"✓ Matrix multiplication ({size}x{size})")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Performance: {tflops:.2f} TFLOPS")
        
        # Expected performance for H200
        expected_tflops = 100  # Rough estimate for FP32
        if tflops < expected_tflops * 0.3:
            print(f"  ⚠️  Performance is lower than expected for H200")
            print(f"      Expected ~{expected_tflops} TFLOPS, got {tflops:.2f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def check_unsloth_config():
    """Check Unsloth configuration."""
    print("\n" + "="*80)
    print("🦥 Unsloth Configuration")
    print("="*80)
    
    try:
        from unsloth import FastLanguageModel
        print("✓ Unsloth installed")
        
        # Check environment variables
        import os
        print("\nEnvironment variables:")
        unsloth_vars = {k: v for k, v in os.environ.items() if 'UNSLOTH' in k or 'FLASH' in k}
        if unsloth_vars:
            for k, v in unsloth_vars.items():
                print(f"  {k} = {v}")
        else:
            print("  No Unsloth-specific env vars set")
            
    except ImportError:
        print("❌ Unsloth not installed")

def main():
    """Main diagnostic function."""
    print("\n🔍 GPU Performance Diagnostic Tool")
    print("="*80)
    
    check_cuda()
    check_flash_attention()
    check_transformers()
    check_gpu_utilization()
    check_unsloth_config()
    test_simple_inference()
    
    print("\n" + "="*80)
    print("✅ Diagnostic Complete")
    print("="*80)
    
    print("\n💡 Recommendations:")
    print("="*80)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("1. ❌ CRITICAL: PyTorch cannot see CUDA")
            print("   - Reinstall PyTorch with CUDA support")
            print("   - Check Docker GPU passthrough (--gpus all)")
    except:
        pass
    
    try:
        import flash_attn
        print("1. ✓ Flash Attention is installed")
    except ImportError:
        print("1. ❌ CRITICAL: Install Flash Attention 2")
        print("   pip install flash-attn --no-build-isolation")
    
    print("\n2. After installing flash-attn, restart your training")
    print("3. Monitor GPU utilization during training:")
    print("   watch -n 1 nvidia-smi")

if __name__ == "__main__":
    main()
