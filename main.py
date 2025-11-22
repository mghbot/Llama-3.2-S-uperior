#!/usr/bin/env python3
"""
Main CLI Interface for Llama 3.2 1B Expert Augmentation System
Provides commands for training, benchmarking, and inference
"""

import argparse
import os
import sys
from typing import Optional

from config import ExpertConfig
from train import fine_tune_augmented_model
from benchmark import run_full_benchmark, compare_models
from surgery import load_augmented_model


def train_command(args):
    """Execute training"""
    config = ExpertConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing=args.gradient_checkpointing,
        quantization_4bit=args.quantization_4bit,
    )

    print("\n" + "="*60)
    print("LLAMA 3.2 1B EXPERT AUGMENTATION - TRAINING")
    print("="*60)
    print(f"Model: {args.model_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Num Experts: {config.num_experts}")
    print(f"Top-K: {config.top_k}")
    print(f"LoRA Rank: {config.lora_r}")
    print("="*60 + "\n")

    fine_tune_augmented_model(
        model_id=args.model_id,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        config=config,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_samples=args.max_samples,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        use_auth_token=args.auth_token,
    )


def benchmark_command(args):
    """Execute benchmarking"""
    print("\n" + "="*60)
    print("LLAMA 3.2 1B EXPERT AUGMENTATION - BENCHMARK")
    print("="*60)
    print(f"Model: {args.model_path}")
    print("="*60 + "\n")

    config = ExpertConfig() if args.is_augmented else None

    run_full_benchmark(
        model_path=args.model_path,
        is_augmented=args.is_augmented,
        config=config,
        use_auth_token=args.auth_token,
        output_file=args.output_file,
    )


def compare_command(args):
    """Execute model comparison"""
    print("\n" + "="*60)
    print("LLAMA 3.2 1B - MODEL COMPARISON")
    print("="*60 + "\n")

    compare_models(
        base_model_path=args.base_model,
        augmented_model_path=args.augmented_model,
        use_auth_token=args.auth_token,
    )


def inference_command(args):
    """Run inference with augmented model"""
    print("\n" + "="*60)
    print("LLAMA 3.2 1B EXPERT AUGMENTATION - INFERENCE")
    print("="*60)
    print(f"Model: {args.model_path}")
    print("="*60 + "\n")

    # Load model
    from transformers import AutoTokenizer

    if args.is_augmented:
        model, config = load_augmented_model(
            args.model_path,
            use_auth_token=args.auth_token
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=args.auth_token
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        token=args.auth_token
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Interactive or single prompt
    if args.prompt:
        prompts = [args.prompt]
    else:
        print("Interactive mode. Type 'quit' to exit.\n")
        prompts = []
        while True:
            try:
                prompt = input("Prompt: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                prompts.append(prompt)
            except KeyboardInterrupt:
                break

    # Generate
    import torch
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated:\n{generated_text}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Llama 3.2 1B Expert Augmentation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python main.py train --dataset wikitext --dataset-config wikitext-2-raw-v1
  python main.py train --dataset HuggingFaceH4/ultrachat_200k --use-wandb

  # Benchmarking
  python main.py benchmark --model-path ./llama-3.2-1b-expert
  python main.py benchmark --model-path meta-llama/Llama-3.2-1B --no-augmented

  # Comparison
  python main.py compare

  # Inference
  python main.py inference --model-path ./llama-3.2-1b-expert --prompt "Hello, world!"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Training command
    train_parser = subparsers.add_parser("train", help="Train augmented model")
    train_parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B", help="Base model ID")
    train_parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    train_parser.add_argument("--dataset-config", default="wikitext-2-raw-v1", help="Dataset config")
    train_parser.add_argument("--output-dir", default="./llama-3.2-1b-expert", help="Output directory")
    train_parser.add_argument("--num-experts", type=int, default=8, help="Number of experts")
    train_parser.add_argument("--top-k", type=int, default=2, help="Top-K experts")
    train_parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    train_parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    train_parser.add_argument("--save-steps", type=int, default=500, help="Save steps")
    train_parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    train_parser.add_argument("--gradient-checkpointing", action="store_true", help="Use gradient checkpointing")
    train_parser.add_argument("--quantization-4bit", action="store_true", help="Use 4-bit quantization")
    train_parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    train_parser.add_argument("--wandb-project", default="llama-expert-augmentation", help="W&B project name")
    train_parser.add_argument("--auth-token", default=None, help="HuggingFace auth token")
    train_parser.set_defaults(func=train_command)

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model")
    benchmark_parser.add_argument("--model-path", default="./llama-3.2-1b-expert", help="Model path")
    benchmark_parser.add_argument("--no-augmented", dest="is_augmented", action="store_false", help="Model is not augmented")
    benchmark_parser.add_argument("--output-file", default=None, help="Output JSON file")
    benchmark_parser.add_argument("--auth-token", default=None, help="HuggingFace auth token")
    benchmark_parser.set_defaults(func=benchmark_command, is_augmented=True)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare base vs augmented")
    compare_parser.add_argument("--base-model", default="meta-llama/Llama-3.2-1B", help="Base model path")
    compare_parser.add_argument("--augmented-model", default="./llama-3.2-1b-expert", help="Augmented model path")
    compare_parser.add_argument("--auth-token", default=None, help="HuggingFace auth token")
    compare_parser.set_defaults(func=compare_command)

    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument("--model-path", default="./llama-3.2-1b-expert", help="Model path")
    inference_parser.add_argument("--prompt", default=None, help="Input prompt (interactive if not provided)")
    inference_parser.add_argument("--max-new-tokens", type=int, default=100, help="Max new tokens")
    inference_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    inference_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p")
    inference_parser.add_argument("--no-sample", dest="do_sample", action="store_false", help="Use greedy decoding")
    inference_parser.add_argument("--no-augmented", dest="is_augmented", action="store_false", help="Model is not augmented")
    inference_parser.add_argument("--auth-token", default=None, help="HuggingFace auth token")
    inference_parser.set_defaults(func=inference_command, do_sample=True, is_augmented=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
