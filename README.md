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
├── wikitext-tokenizer-merges.txt
└── wikitext-tokenizer-vocab.json
```

-   `configs/`: Contains YAML configuration files for training.
-   `inference.py`: Script to run inference with a trained model.
-   `train_tokenizer.py`: Script to train a BPE tokenizer on the WikiText-103 dataset.
-   `train.py`: Script to train the GPT-2 style model.
-   `requirements.txt`: Python dependencies.
-   `wikitext-tokenizer-merges.txt`, `wikitext-tokenizer-vocab.json`: Tokenizer files.
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

### 1. Train the Tokenizer

First, you need to train the tokenizer on the WikiText-103 dataset. The tokenizer files are already provided in the repository, but you can retrain it by running:

```bash
python train_tokenizer.py
```

This will download the dataset and save the tokenizer files (`wikitext-tokenizer-vocab.json` and `wikitext-tokenizer-merges.txt`) in the root directory.

### 2. Train the Model

To train the model, use the `train.py` script with a configuration file.

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

You can modify these files to change hyperparameters, model architecture, dataset, etc.
