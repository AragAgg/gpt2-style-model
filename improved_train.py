import yaml
import argparse
import os
import time
from datetime import timedelta
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
from pathlib import Path


class TokenCounterCallback(TrainerCallback):
    """Callback to track tokens per epoch, stop when target is reached, and display ETA."""
    
    def __init__(self, target_tokens_per_epoch=20_000_000_000, max_seq_length=1024):
        self.target_tokens_per_epoch = target_tokens_per_epoch
        self.max_seq_length = max_seq_length
        self.current_epoch_tokens = 0
        self.total_tokens = 0
        self.current_epoch = 0
        self.epoch_start_time = None
        self.training_start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"🚀 Training Started")
        print(f"   Total epochs planned: {args.num_train_epochs}")
        print(f"   Tokens per epoch target: {self.target_tokens_per_epoch:,}")
        print(f"{'='*80}\n")
        return control
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Calculate tokens processed in this step
        # batch_size * seq_length * world_size (for distributed training)
        # We use max_seq_length as an approximation since we pad to max length
        tokens_this_step = args.per_device_train_batch_size * self.max_seq_length * max(1, args.world_size)
        
        self.current_epoch_tokens += tokens_this_step
        self.total_tokens += tokens_this_step
        
        # Log progress every 100 steps with ETA
        if state.global_step % 100 == 0:
            progress_pct = (self.current_epoch_tokens / self.target_tokens_per_epoch) * 100
            
            # Calculate ETA for current epoch
            if self.epoch_start_time and self.current_epoch_tokens > 0:
                elapsed_time = time.time() - self.epoch_start_time
                tokens_per_second = self.current_epoch_tokens / elapsed_time
                remaining_tokens = self.target_tokens_per_epoch - self.current_epoch_tokens
                eta_seconds = remaining_tokens / tokens_per_second if tokens_per_second > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                
                # Calculate overall training ETA
                overall_elapsed = time.time() - self.training_start_time
                overall_tokens_per_sec = self.total_tokens / overall_elapsed if overall_elapsed > 0 else 0
                total_target_tokens = self.target_tokens_per_epoch * args.num_train_epochs
                remaining_total_tokens = total_target_tokens - self.total_tokens
                overall_eta_seconds = remaining_total_tokens / overall_tokens_per_sec if overall_tokens_per_sec > 0 else 0
                overall_eta_str = str(timedelta(seconds=int(overall_eta_seconds)))
                
                print(f"Epoch {int(state.epoch)}: {self.current_epoch_tokens:,} / {self.target_tokens_per_epoch:,} tokens ({progress_pct:.1f}%) | "
                      f"Epoch ETA: {eta_str} | Overall ETA: {overall_eta_str} | "
                      f"Speed: {tokens_per_second:,.0f} tok/s")
            else:
                print(f"Epoch {int(state.epoch)}: {self.current_epoch_tokens:,} / {self.target_tokens_per_epoch:,} tokens ({progress_pct:.1f}%)")
        
        # Check if we've hit the target for this epoch
        if self.current_epoch_tokens >= self.target_tokens_per_epoch:
            epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
            print(f"\n{'='*80}")
            print(f"✓ Reached {self.current_epoch_tokens:,} tokens in epoch {int(state.epoch)}")
            print(f"  Target was {self.target_tokens_per_epoch:,} tokens")
            print(f"  Epoch duration: {str(timedelta(seconds=int(epoch_time)))}")
            print(f"  Total tokens processed: {self.total_tokens:,}")
            print(f"{'='*80}\n")
            control.should_epoch_stop = True
            
        return control
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch)
        self.current_epoch_tokens = 0
        self.epoch_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"🚀 Starting Epoch {self.current_epoch}")
        print(f"   Target tokens for this epoch: {self.target_tokens_per_epoch:,}")
        print(f"{'='*80}\n")
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        print(f"\n{'='*80}")
        print(f"✓ Completed Epoch {self.current_epoch}")
        print(f"  Tokens processed in this epoch: {self.current_epoch_tokens:,}")
        print(f"  Epoch duration: {str(timedelta(seconds=int(epoch_time)))}")
        print(f"  Total tokens so far: {self.total_tokens:,}")
        print(f"{'='*80}\n")
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.training_start_time if self.training_start_time else 0
        print(f"\n{'='*80}")
        print(f"🎉 Training Completed!")
        print(f"   Total duration: {str(timedelta(seconds=int(total_time)))}")
        print(f"   Total tokens processed: {self.total_tokens:,}")
        avg_speed = self.total_tokens / total_time if total_time > 0 else 0
        print(f"   Average speed: {avg_speed:,.0f} tokens/second")
        print(f"{'='*80}\n")
        return control


class ProfilerCallback(TrainerCallback):
    """Callback to enable PyTorch profiler with TensorBoard export."""
    
    def __init__(self, profiler_output_dir, profile_steps=10, warmup_steps=2, active_steps=5):
        """
        Args:
            profiler_output_dir: Directory to save profiler traces
            profile_steps: Total steps to profile (default: 10)
            warmup_steps: Number of warmup steps (default: 2)
            active_steps: Number of active profiling steps (default: 5)
        """
        self.profiler_output_dir = profiler_output_dir
        self.profile_steps = profile_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.profiler = None
        self.profiling_enabled = False
        
    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(self.profiler_output_dir, exist_ok=True)
        print(f"\n📊 PyTorch Profiler enabled - traces will be saved to: {self.profiler_output_dir}")
        print(f"   Profile config: {self.warmup_steps} warmup + {self.active_steps} active steps")
        print(f"   View with: tensorboard --logdir={self.profiler_output_dir}\n")
        
        # Initialize profiler
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=0,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=1
            ),
            on_trace_ready=tensorboard_trace_handler(self.profiler_output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.profiler.__enter__()
        self.profiling_enabled = True
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.profiling_enabled and self.profiler:
            self.profiler.step()
            
            # Stop profiling if profile_steps is reached. If -1, this condition is never met.
            if self.profile_steps != -1 and state.global_step >= self.profile_steps:
                self.profiler.__exit__(None, None, None)
                self.profiling_enabled = False
                print(f"\n📊 Profiling completed at step {state.global_step}. Check TensorBoard for results.\n")
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.profiling_enabled and self.profiler:
            self.profiler.__exit__(None, None, None)
            self.profiling_enabled = False
        return control


def get_next_run_dir(base_dir="model"):
    """Find the next available run directory (run1, run2, etc.)."""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    run_num = 1
    while True:
        run_dir = base_path / f"run{run_num}"
        if not run_dir.exists():
            return str(run_dir)
        run_num += 1


def main(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_config = config.get('dataset', {})
    model_config = config['model']
    training_config = config['training']

    # Determine the next run directory
    run_dir = get_next_run_dir("model")
    print(f"Using run directory: {run_dir}")
    
    # Update training config with proper paths
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    final_dir = os.path.join(run_dir, "final")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    profiler_dir = os.path.join(run_dir, "profiler")
    training_config['output_dir'] = checkpoint_dir
    training_config['logging_dir'] = tensorboard_dir
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(profiler_dir, exist_ok=True)
    
    # For streaming datasets, we need to calculate max_steps manually
    if training_config.get("num_train_epochs") and not training_config.get("max_steps"):
        tokens_per_epoch = dataset_config.get('tokens_per_epoch', 20_000_000_000)
        
        # Get number of GPUs from accelerate environment
        num_gpus = int(os.environ.get("WORLD_SIZE", 1))
        
        per_device_batch_size = training_config['per_device_train_batch_size']
        grad_accum_steps = training_config['gradient_accumulation_steps']
        seq_length = model_config['n_positions']
        num_epochs = training_config['num_train_epochs']

        # Calculate tokens per optimizer step
        tokens_per_step = per_device_batch_size * num_gpus * grad_accum_steps * seq_length
        
        if tokens_per_step > 0:
            steps_per_epoch = tokens_per_epoch // tokens_per_step
            max_steps = int(steps_per_epoch * num_epochs)
            
            training_config['max_steps'] = max_steps
            del training_config['num_train_epochs']  # max_steps overrides num_train_epochs
            
            print(f"\n{'='*80}")
            print(f"📈 Calculated max_steps for streaming dataset")
            print(f"   Tokens per epoch: {tokens_per_epoch:,}")
            print(f"   Tokens per step: {tokens_per_step:,}")
            print(f"   Steps per epoch: {steps_per_epoch:,}")
            print(f"   Total epochs planned: {num_epochs}")
            print(f"   Calculated max_steps: {max_steps:,}")
            print(f"   Note: `num_train_epochs` is now ignored in favor of `max_steps`.")
            print(f"{'='*80}\n")
    
    # Setup cache directory for datasets
    cache_dir = dataset_config.get('cache_dir', './dataset_cache')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")

    # 1. Load Default GPT-2 Tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dataset with Streaming and Caching
    print("Loading falcon/refinedweb dataset with streaming and caching...")
    
    # Enable HF_DATASETS_CACHE for persistent caching
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    
    dataset = load_dataset(
        "tiiuae/falcon-refinedweb",
        streaming=True,
        split="train",
        cache_dir=cache_dir
    )
    
    # Tokenize function for streaming
    def tokenize_function(examples):
        return tokenizer(
            examples["content"],
            truncation=True,
            max_length=model_config['n_positions'],
            padding='max_length',
            return_tensors=None
        )

    # Apply tokenization to streaming dataset with caching
    print("Tokenizing dataset (cached for subsequent epochs)...")
    
    # Shuffle with a buffer for better randomization while maintaining cache
    buffer_size = dataset_config.get('shuffle_buffer_size', 10_000)
    dataset = dataset.shuffle(seed=42, buffer_size=buffer_size)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["content", "url", "timestamp", "dump", "segment", "image_urls"]
    )
    
    # For streaming datasets, we can't split easily, so we use the same stream for eval
    # Take a small portion for validation
    train_dataset = tokenized_dataset
    
    # Create a separate eval dataset (take first 1000 examples) with caching
    print("Loading validation dataset...")
    eval_dataset = load_dataset(
        "tiiuae/falcon-refinedweb",
        streaming=True,
        split="train",
        cache_dir=cache_dir
    ).take(1000)
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["content", "url", "timestamp", "dump", "segment", "image_urls"]
    )

    # 3. Configure the GPT-2 model
    print("Configuring GPT-2 model...")
    gpt2_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=model_config['n_positions'],
        n_embd=model_config['n_embd'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_rms_norm=True,
        rms_norm_eps=1e-08,
        # Dropout configurations
        resid_pdrop=model_config.get('resid_pdrop', 0.1),
        embd_pdrop=model_config.get('embd_pdrop', 0.1),
        attn_pdrop=model_config.get('attn_pdrop', 0.1),
    )

    # 4. Instantiate the model
    print("Initializing model...")
    model = GPT2LMHeadModel(config=gpt2_config)
    print(f"Model has {model.num_parameters():,} parameters")

    # 5. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 6. Training Arguments
    training_args = TrainingArguments(**training_config)

    # 7. Initialize Callbacks
    token_limit = dataset_config.get('tokens_per_epoch', 20_000_000_000)
    token_callback = TokenCounterCallback(
        target_tokens_per_epoch=token_limit,
        max_seq_length=model_config['n_positions']
    )
    
    # Initialize profiler callback if enabled
    callbacks = [token_callback]
    profiler_config = config.get('profiler', {})
    if profiler_config.get('enabled', True):
        profiler_callback = ProfilerCallback(
            profiler_output_dir=profiler_dir,
            profile_steps=profiler_config.get('profile_steps', 10),
            warmup_steps=profiler_config.get('warmup_steps', 2),
            active_steps=profiler_config.get('active_steps', 5)
        )
        callbacks.append(profiler_callback)

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    # 9. Train the model
    print("Starting training...")
    trainer.train()

    # 10. Save the final model
    print(f"Saving final model to {final_dir}...")
    trainer.save_model(final_dir)
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-2 model from a YAML config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    main(args.config)
