import torch
import argparse
import csv
import json
import os
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from inference import generate_response

def load_model_and_tokenizer(model_path):
    """
    Load the model and tokenizer from the given path.
    
    Args:
        model_path: Path to the model checkpoint directory
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        # Set pad token to eos token if it's not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_questions_from_json(json_path):
    """
    Load questions from a JSON file.
    Expected format: [{"question": "...", "max_tokens": 50}, ...]
    
    Args:
        json_path: Path to JSON file containing questions
    
    Returns:
        list: List of dictionaries with 'question' and 'max_tokens' keys
    """
    with open(json_path, 'r') as f:
        questions = json.load(f)
    return questions

def load_questions_from_csv(csv_path):
    """
    Load questions from a CSV file.
    Expected format: question,max_tokens
    
    Args:
        csv_path: Path to CSV file containing questions
    
    Returns:
        list: List of dictionaries with 'question' and 'max_tokens' keys
    """
    questions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'question': row['question'],
                'max_tokens': int(row['max_tokens'])
            })
    return questions

def run_evaluation(model, tokenizer, questions, temperature=1.0, repetition_penalty=1.2, 
                  top_k=50, top_p=0.9):
    """
    Run evaluation on a list of questions.
    
    Args:
        model: The GPT-2 model
        tokenizer: The tokenizer
        questions: List of dictionaries with 'question' and 'max_tokens' keys
        temperature: Sampling temperature
        repetition_penalty: Penalty for repetition
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
    
    Returns:
        list: List of results with question, max_tokens, and response
    """
    results = []
    
    print(f"\nRunning evaluation on {len(questions)} questions...")
    for i, q in enumerate(questions, 1):
        question = q['question']
        max_tokens = q['max_tokens']
        
        print(f"[{i}/{len(questions)}] Processing: {question[:50]}...")
        
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            max_new_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p
        )
        
        results.append({
            'question': question,
            'max_tokens': max_tokens,
            'response': response
        })
        
    print("Evaluation complete!")
    return results

def save_results_to_csv(results, output_path):
    """
    Save evaluation results to a CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question', 'max_tokens', 'response']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to: {output_path}")

def extract_run_and_checkpoint(model_path):
    """
    Extract the run name and checkpoint name from the model path.
    
    Args:
        model_path: Path to the model directory
    
    Returns:
        tuple: (run_name, checkpoint_name)
    
    Examples:
        "./model/run15/checkpoints/checkpoint-5000" -> ("run15", "checkpoint-5000")
        "./model/run1/final" -> ("run1", "final")
        "/workspace/model/run3/checkpoints/checkpoint-2000" -> ("run3", "checkpoint-2000")
    """
    # Normalize the path
    path_parts = os.path.normpath(model_path).split(os.sep)
    
    run_name = None
    checkpoint_name = None
    
    # Find the run name (looks like "runN")
    for part in path_parts:
        if part.startswith("run") and len(part) > 3:
            # Check if the rest is a number
            try:
                int(part[3:])
                run_name = part
                break
            except ValueError:
                continue
    
    # The checkpoint name is the last directory in the path
    checkpoint_name = path_parts[-1]
    
    # If run name not found, use "unknown_run"
    if run_name is None:
        run_name = "unknown_run"
    
    return run_name, checkpoint_name

def main():
    parser = argparse.ArgumentParser(
        description="Automated evaluation script for GPT-2 style models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with a benchmark name
  python eval.py --model_path ./model/run15/checkpoints/checkpoint-5000 --questions questions.json --benchmark_name my_benchmark
  
  # Customize generation parameters and specify output directory
  python eval.py --model_path ./model/run1/checkpoints/checkpoint-1000 --questions questions.json --benchmark_name another_benchmark --temperature 0.8 --output_dir ./custom_evals
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint directory."
    )
    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to questions file (JSON or CSV format)."
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        required=True,
        help="Required name for the benchmark run (used in output filename)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results. Defaults to '<model_path>/evals/'."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)."
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (default: 1.2)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter (default: 0.9)."
    )
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load questions
    questions_path = args.questions
    if questions_path.endswith('.json'):
        questions = load_questions_from_json(questions_path)
    elif questions_path.endswith('.csv'):
        questions = load_questions_from_csv(questions_path)
    else:
        raise ValueError("Questions file must be either .json or .csv format")
    
    print(f"Loaded {len(questions)} questions from {questions_path}")
    
    # Run evaluation
    results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Extract run name and checkpoint from model path for summary
    run_name, checkpoint_name = extract_run_and_checkpoint(args.model_path)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_path, 'evals')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    output_filename = f"eval_{args.benchmark_name}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save results
    save_results_to_csv(results, output_path)
    
    print(f"\nEvaluation Summary:")
    print(f"  Model: {args.model_path}")
    print(f"  Run: {run_name}")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Benchmark: {args.benchmark_name}")
    print(f"  Questions: {len(questions)}")
    print(f"  Results: {output_path}")

if __name__ == "__main__":
    main()

