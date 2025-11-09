# GPT-2 Style Model on WikiText-103

This project contains a GPT-2 style transformer model trained on the WikiText-103 dataset. It includes scripts for training a tokenizer, training the model, and configuration files for different training setups.

## Project Structure

```
gpt2-style-model/
├── configs/
│   ├── full_training_config.yaml
│   └── test_pipeline_config.yaml
├── gpt2-overfit-test/
├── inference.py
├── train_tokenizer.py
├── train.py
├── requirements.txt
├── gpt2-tokenizer-merges.txt
└── gpt2-tokenizer-vocab.json
```

-   `configs/`: Contains YAML configuration files for training.
-   `inference.py`: Script to run inference with a trained model.
-   `train_tokenizer.py`: Script to train a BPE tokenizer on the WikiText-103 dataset.
-   `improved_train_tokeniser.py`: Script to download and setup the default GPT-2 BPE tokenizer.
-   `train.py`: Script to train the GPT-2 style model.
-   `requirements.txt`: Python dependencies.
-   `gpt2-tokenizer-merges.txt`, `gpt2-tokenizer-vocab.json`: GPT-2 pretrained tokenizer files (original GPT-2 embeddings).
-   `gpt2-overfit-test/`: Directory where the overfit test model is saved.

## Getting Started

### Prerequisites

-   Conda
-   Python 3.7+

### Installation

1.  Clone the repository.
2.  Create and activate a conda environment:

    ```bash
    conda create -n gpt2-style python=3.8
    conda activate gpt2-style
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Login to Weights & Biases (for experiment tracking):

    ```bash
    wandb login
    ```

    You'll need to provide your API key. If you don't have a wandb account, you can create one at [wandb.ai](https://wandb.ai).

## Usage

### 1. Setup the Tokenizer

You have two options for setting up the tokenizer:

#### Option A: Use Default GPT-2 Tokenizer (Recommended)

To download and setup the default GPT-2 BPE tokenizer:

```bash
python improved_train_tokeniser.py
```

This will download the official GPT-2 tokenizer from HuggingFace and save the tokenizer files (`gpt2-tokenizer-vocab.json` and `gpt2-tokenizer-merges.txt`) in the root directory. These are the **original GPT-2 embeddings**, not a custom-trained tokenizer.

#### Option B: Train a Custom Tokenizer

To train a custom tokenizer on the WikiText-103 dataset:

```bash
python train_tokenizer.py
```

This will download the dataset and train a custom BPE tokenizer on WikiText-103, saving the tokenizer files. Note: This creates a custom tokenizer, not the original GPT-2 tokenizer.

### 2. Train the Model

You can use either `train.py` for basic training or `improved_train.py` for advanced streaming training.

#### Option A: Improved Training (Recommended for Large-Scale)

The `improved_train.py` script provides advanced features:
- **Streaming dataset loading** for memory-efficient training on large datasets
- **Dataset caching**: Downloaded data is cached locally, preventing re-downloads across epochs
- **Automatic run management**: Saves models to `model/run1/`, `model/run2/`, etc.
- **Token-based epoch control**: Each epoch stops after processing >20B tokens (configurable)
- **ETA logging**: Real-time estimates for epoch and overall training completion
- **Performance metrics**: Tokens/second speed tracking throughout training
- **PyTorch Profiler**: Automatic profiling with TensorBoard visualization
- **TensorBoard integration**: Comprehensive training visualization
- **Default GPT-2 tokenizer**: Uses pretrained tokenizer (no custom vocab needed)
- **Falcon RefinedWeb dataset**: High-quality web text data

**⚠️ Before Training**: 

1. **Run verification script** (recommended):
   ```bash
   python verify_setup.py
   ```
   This checks GPU, disk space, packages, and network connectivity.

2. **Or review manual checklist**: See [`PRE_TRAINING_CHECKLIST.md`](PRE_TRAINING_CHECKLIST.md)

3. **Start training**:
   ```bash
   python improved_train.py --config configs/improved_training_config.yaml
   ```

You can adjust the `tokens_per_epoch` setting in the config file to control when each epoch ends.

**Output Structure:**
```
model/runN/
├── checkpoints/     # Saved checkpoints during training
├── final/          # Final trained model
├── tensorboard/    # TensorBoard training logs
└── profiler/       # PyTorch profiler traces
```

**Monitor Training:**
```bash
# View training logs in real-time
tensorboard --logdir=model/run1/tensorboard --port=6006

# View profiler traces (performance analysis)
tensorboard --logdir=model/run1/profiler --port=6007
```

Where N is automatically incremented (run1, run2, run3, etc.).

#### Option B: Standard Training

#### Overfit Test

It's recommended to run a small test to make sure the training pipeline is working correctly. This will train a small model on a tiny subset of the data to check for overfitting.

```bash
accelerate launch --config_file configs/accelerate_config.yaml train.py --config configs/test_pipeline_config.yaml
```

This command uses the `test_pipeline_config.yaml` to configure the model and training parameters for a small-scale run.

This should result in a very low validation loss after a few epochs, indicating that the model is able to memorize the small dataset.

#### Full Training

To run the full training on the WikiText-103 dataset, use the `full_training_config.yaml` file.

```bash
accelerate launch train.py --config configs/full_training_config.yaml
```

This will train a larger model on the entire dataset. Training progress is logged to Weights & Biases.

### 3. Run Inference

To generate text with the trained model, use the `inference.py` script.

```bash
python inference.py --model_path ./gpt2-wikitext-full-final --prompt "Once upon a time"
```

You can customize the generation parameters:

-   `--model_path`: Path to the saved model directory.
-   `--prompt`: The text prompt to start generation from.
-   `--max_length`: Maximum length of the generated text.
-   `--temperature`: Controls randomness.
-   `--top_k`: Top-K filtering.
-   `--top_p`: Nucleus sampling.

## Configuration

The training process is controlled by YAML configuration files in the `configs/` directory.

-   `test_pipeline_config.yaml`: Configuration for a small-scale overfitting test.
-   `full_training_config.yaml`: Configuration for a full training run on the WikiText-103 dataset.
-   `improved_training_config.yaml`: Configuration for streaming training on Falcon RefinedWeb with automatic run management and token-based epoch control.

You can modify these files to change hyperparameters, model architecture, dataset, etc.

### Config Options for improved_train.py

The `improved_training_config.yaml` supports these key options:

- `dataset.tokens_per_epoch`: Number of tokens to process per epoch (default: 20B)
- `dataset.cache_dir`: Directory for caching downloaded dataset (default: `./dataset_cache`)
- `dataset.shuffle_buffer_size`: Buffer size for shuffling streaming data (default: 10,000)
- `profiler.enabled`: Enable/disable PyTorch profiler (default: true)
- `profiler.profile_steps`: Number of steps to profile (default: 10)
- `profiler.warmup_steps`: Warmup steps before profiling (default: 2)
- `profiler.active_steps`: Active profiling steps (default: 5)
- `model.*`: Standard GPT-2 architecture parameters
- `training.*`: Standard HuggingFace Trainer arguments

Note: The `output_dir` in the config will be overridden by the automatic run directory system (`model/runN/checkpoints/`).

#### Dataset Caching

The streaming dataset is automatically cached in the `dataset_cache/` directory. This means:
- First epoch: Data is downloaded from HuggingFace and cached locally
- Subsequent epochs: Data is read from the local cache (much faster!)
- The cache persists across training runs
- Delete `dataset_cache/` if you want to re-download fresh data

#### PyTorch Profiler & Performance Analysis

The script includes built-in PyTorch profiling:
- **Automatic profiling**: First 10 steps are profiled by default (minimal overhead)
- **TensorBoard export**: Traces saved to `model/runN/profiler/`
- **Metrics captured**: CPU/GPU usage, memory allocation, operation timing
- **View results**: `tensorboard --logdir=model/runN/profiler`

To disable profiling or adjust settings, edit `improved_training_config.yaml`:
```yaml
profiler:
  enabled: false  # Disable profiling
```

#### Training Progress & ETA

During training, you'll see detailed progress logs:
```
Epoch 0: 327,680,000 / 20,000,000,000 tokens (1.6%) | 
  Epoch ETA: 5:23:15 | Overall ETA: 16:09:45 | Speed: 1,234,567 tok/s
```

- **Epoch ETA**: Time remaining for current epoch
- **Overall ETA**: Time remaining for all training
- **Speed**: Tokens processed per second (throughput metric)
