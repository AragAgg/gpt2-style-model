# Quick Start Guide

## 🚀 Start Training in 3 Steps

### Step 1: Install Dependencies
```bash
conda activate gpt2-style  # or your environment name
pip install -r requirements.txt
```

### Step 2: Verify Setup
```bash
# Run automated verification
python verify_setup.py

# Or manual GPU check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Step 3: Start Training
```bash
python improved_train.py --config configs/improved_training_config.yaml
```

## 📊 Monitor Training

### Open TensorBoard (in separate terminal)
```bash
# Training metrics
tensorboard --logdir=model/run1/tensorboard --port=6006

# Performance profiler
tensorboard --logdir=model/run1/profiler --port=6007
```

Then open:
- http://localhost:6006 - Training dashboard
- http://localhost:6007 - Performance profiler

### Watch GPU Usage
```bash
watch -n 1 nvidia-smi
```

## ⚙️ Common Adjustments

### Reduce Memory Usage (if OOM)
Edit `configs/improved_training_config.yaml`:
```yaml
training:
  per_device_train_batch_size: 8  # Reduce from 32
  gradient_accumulation_steps: 16  # Increase from 4
```

### Faster Testing
```yaml
dataset:
  tokens_per_epoch: 1000000  # 1M tokens for quick test
training:
  num_train_epochs: 1
```

### Disable Profiler (slight speedup)
```yaml
profiler:
  enabled: false
```

## 📁 Output Structure

After training starts:
```
model/run1/
├── checkpoints/     # Saved every 5000 steps
├── final/          # Final model at end
├── tensorboard/    # Training logs
└── profiler/       # Performance traces
```

## ⚠️ Important Notes

1. **First run**: Dataset download takes 1-3 hours (cached after)
2. **Disk space**: Need ~300GB+ free
3. **Training time**: ~6-12 hours per epoch on H100 (20B tokens)
4. **Internet**: Required for first epoch only

## 🔍 Need More Details?

See [`PRE_TRAINING_CHECKLIST.md`](PRE_TRAINING_CHECKLIST.md) for comprehensive setup guide.

## 🆘 Quick Troubleshooting

**Out of memory?**
```yaml
per_device_train_batch_size: 8  # or 4
gradient_accumulation_steps: 16  # or 32
```

**Download too slow?**
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**Resume from checkpoint?**
Trainer auto-resumes from latest checkpoint in output_dir.

**Stop training gracefully?**
`Ctrl+C` once - trainer will save checkpoint before exiting.

