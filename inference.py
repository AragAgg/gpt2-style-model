import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, 
                     repetition_penalty=1.2, top_k=50, top_p=0.9, do_sample=True):
    """
    Generate a response for a given prompt using the model.
    
    Args:
        model: The GPT-2 model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of new tokens to generate (excluding prompt)
        temperature: Sampling temperature
        repetition_penalty: Penalty for repetition
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        do_sample: Whether to use sampling
    
    Returns:
        Generated text (without the input prompt)
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            num_return_sequences=1
        )

    # The generated output includes the prompt, so we slice it off.
    generated_text = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return generated_text

def interactive_chat(model, tokenizer):
    """
    Handles the interactive chat loop, generating responses from the model.
    """
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            break

        generated_text = generate_response(model, tokenizer, prompt, max_new_tokens=50)
        print("Bot:", generated_text)

def main():
    """
    Main function to load the model and start the interactive chat.
    """
    parser = argparse.ArgumentParser(description="Interactive chat with a GPT-2 style model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint directory."
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        # Set pad token to eos token if it's not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Chatbot initialized. Type 'exit' or 'quit' to end the conversation.")
    interactive_chat(model, tokenizer)

if __name__ == "__main__":
    main()
