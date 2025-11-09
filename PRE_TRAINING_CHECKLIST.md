# Pre-Training Checklist

Before starting training with `improved_train.py`, please verify the following:

## ✅ Environment Setup

### 1. **Python Environment**
```bash
# Check conda is installed
conda --version

# Verify environment is activated
conda activate gpt2-style  # or your environment name
```

### 2. **Install/Update Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# Verify key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import tensorboard; print('TensorBoard: OK')"
```

### 3. **GPU Availability**
```bash
# Check CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## ✅ Configuration Review

### 4. **Check Training Config** (`configs/improved_training_config.yaml`)
- [ ] `tokens_per_epoch`: Set to 20B (or adjust based on your needs)
- [ ] `cache_dir`: Ensure you have enough disk space for cached data
- [ ] `per_device_train_batch_size`: Adjust based on GPU memory
- [ ] `gradient_accumulation_steps`: Adjust to fit effective batch size
- [ ] `num_train_epochs`: Set desired number of epochs
- [ ] `save_steps`: Adjust checkpoint frequency
- [ ] `bf16` or `fp16`: Enable based on GPU capability (bf16 for modern GPUs)

### 5. **Model Size Estimation**
Current config (12 layers, 768 dim, 12 heads):
- **~124M parameters**
- **GPU Memory**: ~4-6GB for model + ~2-8GB per batch (depends on batch size & sequence length)
- **Recommended**: 16GB+ GPU for batch_size=32, seq_len=1024

Adjust if needed:
```yaml
model:
  n_layer: 12      # Reduce for smaller model
  n_embd: 768      # Reduce for smaller model
  n_positions: 1024  # Reduce for less memory usage
```

## ✅ Storage & Disk Space

### 6. **Check Available Disk Space**
```bash
# Check current disk usage
df -h .

# Estimate required space:
# - Dataset cache: ~50-200GB (Falcon RefinedWeb)
# - Model checkpoints: ~500MB per checkpoint × num_checkpoints
# - TensorBoard logs: ~1-5GB
# - Profiler traces: ~500MB-2GB
# Recommended: 300GB+ free space
```

### 7. **Directory Structure**
The script will automatically create:
```
model/
├── run1/
│   ├── checkpoints/     # Model checkpoints
│   ├── final/          # Final trained model
│   ├── tensorboard/    # TensorBoard logs
│   └── profiler/       # PyTorch profiler traces
dataset_cache/          # Cached Falcon RefinedWeb data
```

## ✅ Monitoring Setup

### 8. **Weights & Biases (Optional but Recommended)**
```bash
# Login to W&B
wandb login

# Or disable W&B in config
# report_to: ["tensorboard"]  # Remove "wandb"
```

### 9. **TensorBoard Setup**
```bash
# Test TensorBoard installation
tensorboard --version

# During training, in a separate terminal:
tensorboard --logdir=model/run1/tensorboard --port=6006

# Then open: http://localhost:6006
```

## ✅ Network & Data Access

### 10. **HuggingFace Hub Access**
```bash
# Test connection to HuggingFace
python -c "from datasets import load_dataset; print('HF Hub: OK')"

# Optional: Login for private datasets
huggingface-cli login

# Check dataset access
python -c "from datasets import load_dataset; ds = load_dataset('tiiuae/falcon-refinedweb', streaming=True, split='train'); print('Dataset accessible:', next(iter(ds)))"
```

### 11. **Internet Connection**
- **First epoch**: Requires stable internet to download dataset
- **Subsequent epochs**: Can run offline (uses cached data)
- **Tip**: Consider running overnight for first epoch to complete download

## ✅ Performance Optimization

### 12. **PyTorch Compilation (Optional)**
If using PyTorch 2.0+:
```yaml
training:
  torch_compile: true  # Enable compilation for speedup
```

### 13. **Profiler Settings**
Adjust profiler in config:
```yaml
profiler:
  enabled: true        # Set false to disable profiling
  profile_steps: 10    # Profiles first 10 steps only
```

## ✅ Final Checks

### 14. **Test Run (Highly Recommended)**
```bash
# Create a test config with small values
cp configs/improved_training_config.yaml configs/test_run.yaml

# Edit test_run.yaml:
# - tokens_per_epoch: 1000000  # 1M tokens for quick test
# - num_train_epochs: 1
# - save_steps: 100

# Run test
python improved_train.py --config configs/test_run.yaml

# This should complete in a few minutes
```

### 15. **Monitor System Resources**
Open in separate terminals:
```bash
# GPU monitoring
watch -n 1 nvidia-smi

# System monitoring
htop

# Disk usage
watch -n 60 df -h
```

## 🚀 Start Training

Once all checks pass:

```bash
# Start training
python improved_train.py --config configs/improved_training_config.yaml

# Or with nohup to run in background:
nohup python improved_train.py --config configs/improved_training_config.yaml > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

## 📊 During Training - What to Expect

### Expected Output
```
Using run directory: model/run1
Using cache directory: ./dataset_cache

🚀 Training Started
   Total epochs planned: 3
   Tokens per epoch target: 20,000,000,000

📊 PyTorch Profiler enabled - traces will be saved to: model/run1/profiler
   View with: tensorboard --logdir=model/run1/profiler

🚀 Starting Epoch 0
   Target tokens for this epoch: 20,000,000,000

Epoch 0: 32,768,000 / 20,000,000,000 tokens (0.2%) | Epoch ETA: 5:23:15 | Overall ETA: 16:09:45 | Speed: 1,234,567 tok/s
```

### ETA Information
- **Epoch ETA**: Time until current epoch completes
- **Overall ETA**: Time until all training completes
- **Speed**: Tokens processed per second
- Updates every 100 steps

### TensorBoard Metrics
- Training loss
- Learning rate
- GPU memory usage
- Profiler traces (first 10 steps)

## ⚠️ Troubleshooting

### Out of Memory (OOM)
```yaml
# Reduce batch size
per_device_train_batch_size: 16  # or 8

# Increase gradient accumulation
gradient_accumulation_steps: 8  # or 16

# Reduce sequence length
model:
  n_positions: 512  # or 256
```

### Slow Training
- Check GPU utilization with `nvidia-smi`
- Enable `torch_compile: true`
- Increase `dataloader_num_workers`
- Check disk I/O isn't bottleneck

### Dataset Download Stalls
- Check internet connection
- Try: `export HF_HUB_ENABLE_HF_TRANSFER=1`
- Clear cache: `rm -rf dataset_cache/`

## 📝 Notes

- **First run**: Dataset download may take 1-3 hours depending on connection
- **20B tokens/epoch**: ~6-12 hours on a single H100 GPU (estimate)
- **Checkpoints**: Saved every 5000 steps by default
- **Profiler**: Only profiles first 10 steps to minimize overhead
- **Auto-resume**: Trainer supports resuming from checkpoints automatically

## 🎯 Post-Training

After training completes:
```bash
# Final model location
ls -lh model/run1/final/

# View TensorBoard logs
tensorboard --logdir=model/run1/tensorboard

# View profiler traces
tensorboard --logdir=model/run1/profiler
```

---

**Good luck with your training! 🚀**

