#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    errors = []

    # Test config
    try:
        from config import ExpertConfig
        config = ExpertConfig()
        print("✓ config.py imports successfully")
        print(f"  - Default num_experts: {config.num_experts}")
        print(f"  - Default top_k: {config.top_k}")
    except Exception as e:
        errors.append(f"✗ config.py failed: {e}")
        print(errors[-1])

    # Test modules
    try:
        from modules import NoisyTopKGating, ExpertFFN, MoEAdapter, KVCacheCompressor
        print("✓ modules.py imports successfully")
        print(f"  - NoisyTopKGating: {NoisyTopKGating}")
        print(f"  - ExpertFFN: {ExpertFFN}")
        print(f"  - MoEAdapter: {MoEAdapter}")
        print(f"  - KVCacheCompressor: {KVCacheCompressor}")
    except Exception as e:
        errors.append(f"✗ modules.py failed: {e}")
        print(errors[-1])

    # Test surgery
    try:
        from surgery import load_augmented_model, inject_experts_into_llama, collect_routing_losses
        print("✓ surgery.py imports successfully")
        print(f"  - load_augmented_model: {load_augmented_model}")
        print(f"  - inject_experts_into_llama: {inject_experts_into_llama}")
    except Exception as e:
        errors.append(f"✗ surgery.py failed: {e}")
        print(errors[-1])

    # Test train
    try:
        from train import fine_tune_augmented_model, ExpertAugmentedTrainer
        print("✓ train.py imports successfully")
        print(f"  - fine_tune_augmented_model: {fine_tune_augmented_model}")
        print(f"  - ExpertAugmentedTrainer: {ExpertAugmentedTrainer}")
    except Exception as e:
        errors.append(f"✗ train.py failed: {e}")
        print(errors[-1])

    # Test benchmark
    try:
        from benchmark import ExpertModelBenchmark, run_full_benchmark, compare_models
        print("✓ benchmark.py imports successfully")
        print(f"  - ExpertModelBenchmark: {ExpertModelBenchmark}")
        print(f"  - run_full_benchmark: {run_full_benchmark}")
    except Exception as e:
        errors.append(f"✗ benchmark.py failed: {e}")
        print(errors[-1])

    # Test main
    try:
        import main
        print("✓ main.py imports successfully")
    except Exception as e:
        errors.append(f"✗ main.py failed: {e}")
        print(errors[-1])

    # Test PyTorch
    try:
        import torch
        print(f"\n✓ PyTorch {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - Device count: {torch.cuda.device_count()}")
    except Exception as e:
        errors.append(f"✗ PyTorch failed: {e}")
        print(errors[-1])

    # Test Transformers
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except Exception as e:
        errors.append(f"✗ Transformers failed: {e}")
        print(errors[-1])

    # Summary
    print("\n" + "="*60)
    if errors:
        print(f"FAILED: {len(errors)} errors found")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("SUCCESS: All imports working correctly!")
        return True
    print("="*60)


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
