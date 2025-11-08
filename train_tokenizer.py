from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# Instantiate a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customized training
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

tokenizer.train_from_iterator(batch_iterator(), vocab_size=52000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model(".", "wikitext-tokenizer")
