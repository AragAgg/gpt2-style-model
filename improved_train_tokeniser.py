from transformers import GPT2Tokenizer
import os
import shutil
import sys

vocab_dest = "./gpt2-tokenizer-vocab.json"
merges_dest = "./gpt2-tokenizer-merges.txt"

# Check if the tokenizer files already exist
if os.path.exists(vocab_dest) and os.path.exists(merges_dest):
    print("Tokenizer files already exist.")
    print(f"  - vocab.json: {os.path.abspath(vocab_dest)}")
    print(f"  - merges.txt: {os.path.abspath(merges_dest)}")
    print("\nTokenizer ready to use! (no download necessary)")
    sys.exit(0)

# Download and load the default GPT-2 tokenizer
print("Downloading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Save the tokenizer files to a temporary directory
temp_dir = "./gpt2-tokenizer-temp"
print("Saving tokenizer files...")
tokenizer.save_pretrained(temp_dir)

# Copy files to the expected names (matching config file expectations)
vocab_source = os.path.join(temp_dir, "vocab.json")
merges_source = os.path.join(temp_dir, "merges.txt")

shutil.copy(vocab_source, vocab_dest)
shutil.copy(merges_source, merges_dest)

# Clean up temporary directory
shutil.rmtree(temp_dir)

print("GPT-2 tokenizer files saved:")
print(f"  - vocab.json: {os.path.abspath(vocab_dest)}")
print(f"  - merges.txt: {os.path.abspath(merges_dest)}")
print("\nTokenizer ready to use!")
