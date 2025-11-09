# Changelog

- Installed Miniconda for environment management.
- Added improved_train_tokeniser.py script to download and setup default GPT-2 BPE tokenizer.
- Renamed tokenizer files to gpt2-tokenizer-* to clarify they use original GPT-2 embeddings, not custom-trained.
- Updated improved_train.py: removed chunking logic, uses default GPT-2 tokenizer, saves to model/runN/checkpoints and model/runN/final, implements 20B token per-epoch limit, uses streaming for falcon/refinedweb dataset.
- Added improved_training_config.yaml for use with improved_train.py script.
- Implemented dataset caching for streaming mode to avoid re-downloading data across epochs.
- Added ETA logging with epoch and overall training time estimates, tokens/second speed tracking.
- Integrated PyTorch profiler with TensorBoard export for performance analysis.
- Added tensorboard to requirements.txt.
- Created PRE_TRAINING_CHECKLIST.md with comprehensive setup verification steps.
- Created QUICK_START.md for rapid training startup.
- Added verify_setup.py automated verification script to check all prerequisites before training.
- Updated improved_training_config.yaml to 1B model (n_embd=2048, n_layer=22, n_head=16) with 4096 context length, dropout configs (resid/attn=0.1, embd=0.05), cosine LR schedule, optimized for 4x H100 GPUs with gradient checkpointing, torch_compile, and fused AdamW.
- Created configs/accelerate_config.yaml for multi-GPU training with bf16 precision.
- Fixed evaluation_strategy parameter to eval_strategy in improved_training_config.yaml for compatibility with newer transformers versions.
- Added logic to improved_train.py to automatically calculate `max_steps` for streaming datasets, fixing ValueError with learning rate scheduler.
- Fixed profiler error by setting default `profile_steps` in config and correcting profiler stop logic in `improved_train.py`.
