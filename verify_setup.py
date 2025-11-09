#!/usr/bin/env python3
"""
Pre-training verification script
Run this before starting training to check your setup
"""

import sys
import os

def check_imports():
    """Check all required packages are installed"""
    print("🔍 Checking Python packages...")
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'yaml': 'PyYAML',
        'tensorboard': 'TensorBoard',
        'wandb': 'Weights & Biases'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\n🔍 Checking GPU/CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Check memory
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU 0 memory: {mem:.1f} GB")
            
            if mem < 12:
                print(f"  ⚠️  Warning: GPU memory < 12GB. Reduce batch size in config.")
            return True
        else:
            print("  ✗ CUDA not available")
            print("  ⚠️  Training will be VERY slow on CPU")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False

def check_disk_space():
    """Check available disk space"""
    print("\n🔍 Checking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (2**30)
        print(f"  ✓ Free space: {free_gb:.1f} GB")
        
        if free_gb < 100:
            print(f"  ⚠️  Warning: Less than 100GB free")
            print(f"     Recommended: 300GB+ for full training")
            return False
        elif free_gb < 300:
            print(f"  ⚠️  Warning: Less than 300GB free")
            print(f"     May need to reduce checkpoint frequency")
        return True
    except Exception as e:
        print(f"  ✗ Error checking disk: {e}")
        return False

def check_config():
    """Check if config file exists and is valid"""
    print("\n🔍 Checking configuration...")
    config_path = "configs/improved_training_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"  ✗ Config file not found: {config_path}")
        return False
    
    print(f"  ✓ Config file exists: {config_path}")
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key settings
        batch_size = config.get('training', {}).get('per_device_train_batch_size', 0)
        print(f"  ✓ Batch size: {batch_size}")
        
        tokens = config.get('dataset', {}).get('tokens_per_epoch', 0)
        print(f"  ✓ Tokens per epoch: {tokens:,}")
        
        epochs = config.get('training', {}).get('num_train_epochs', 0)
        print(f"  ✓ Epochs: {epochs}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error reading config: {e}")
        return False

def check_network():
    """Check network connectivity to HuggingFace"""
    print("\n🔍 Checking network connectivity...")
    try:
        from datasets import load_dataset
        # Try to access the dataset (doesn't download, just checks access)
        print("  ⏳ Testing HuggingFace Hub access...")
        ds = load_dataset("tiiuae/falcon-refinedweb", streaming=True, split="train")
        next(iter(ds))  # Try to get first example
        print("  ✓ HuggingFace Hub accessible")
        print("  ✓ falcon-refinedweb dataset accessible")
        return True
    except Exception as e:
        print(f"  ✗ Network/dataset access error: {e}")
        print(f"  ⚠️  Check internet connection")
        return False

def check_tokenizer():
    """Check GPT-2 tokenizer can be loaded"""
    print("\n🔍 Checking tokenizer...")
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print(f"  ✓ GPT-2 tokenizer loaded")
        print(f"  ✓ Vocab size: {len(tokenizer)}")
        return True
    except Exception as e:
        print(f"  ✗ Error loading tokenizer: {e}")
        return False

def estimate_training_time():
    """Estimate training time"""
    print("\n⏱️  Training Time Estimates:")
    print("  (Based on H100 GPU, adjust for your hardware)")
    print()
    print("  Per Epoch (~20B tokens):")
    print("    - H100 GPU: ~6-12 hours")
    print("    - A100 GPU: ~12-24 hours")
    print("    - V100 GPU: ~24-48 hours")
    print()
    print("  For 3 epochs: multiply above by 3")
    print("  First epoch includes dataset download time (+1-3 hours)")

def main():
    print("="*80)
    print("Pre-Training Setup Verification")
    print("="*80)
    print()
    
    checks = [
        ("Packages", check_imports),
        ("GPU/CUDA", check_cuda),
        ("Disk Space", check_disk_space),
        ("Config File", check_config),
        ("Tokenizer", check_tokenizer),
        ("Network", check_network),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results[name] = False
    
    estimate_training_time()
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    print(f"Passed: {passed}/{total} checks")
    
    if passed == total:
        print("\n✅ All checks passed! Ready to start training.")
        print("\n🚀 Start training with:")
        print("   python improved_train.py --config configs/improved_training_config.yaml")
        print("\n📊 Monitor with:")
        print("   tensorboard --logdir=model/run1/tensorboard --port=6006")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review and fix issues above.")
        print("   See PRE_TRAINING_CHECKLIST.md for detailed instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

