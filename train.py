import yaml
import argparse
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

def main(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tokenizer_config = config['tokenizer']
    dataset_config = config['dataset']
    model_config = config['model']
    training_config = config['training']

    # 1. Load Tokenizer
    tokenizer = GPT2Tokenizer(
        vocab_file=tokenizer_config['vocab_file'],
        merges_file=tokenizer_config['merges_file']
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and Process Dataset
    dataset = load_dataset(dataset_config['name'], dataset_config['config_name'])

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=dataset_config['num_proc'],
        remove_columns=["text"]
    )

    block_size = model_config['n_positions']

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=dataset_config['num_proc'],
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    if dataset_config.get('use_subset', False):
        if dataset_config.get('eval_from_train', False):
            # For the overfit test, train and eval on the same small slice of data to check for memorization
            train_size = dataset_config['subset_size']['train']
            eval_size = dataset_config['subset_size']['validation']
            
            train_dataset = tokenized_datasets["train"].select(range(train_size))
            # The validation set is a SUBSET of the training set
            eval_dataset = train_dataset.select(range(eval_size))
        else:
            train_dataset = train_dataset.select(range(dataset_config['subset_size']['train']))
            eval_dataset = eval_dataset.select(range(dataset_config['subset_size']['validation']))


    # 3. Configure the GPT-2 model
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
    )

    # 4. Instantiate the model
    model = GPT2LMHeadModel(config=gpt2_config)

    # 5. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 6. Training Arguments
    training_args = TrainingArguments(**training_config)

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 8. Train the model
    trainer.train()

    # 9. Save the final model
    final_model_path = training_config['output_dir'] + "-final"
    trainer.save_model(final_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-2 model from a YAML config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    main(args.config)
