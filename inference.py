import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def interactive_chat(model, tokenizer):
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            break

        inputs = tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=100,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=1
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Bot:", generated_text)

def main():
    model_path = "./gpt2-wikitext-full-final"
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    print("Chatbot initialized. Type 'exit' or 'quit' to end the conversation.")
    interactive_chat(model, tokenizer)

if __name__ == "__main__":
    main()
