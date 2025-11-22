"""
Comprehensive Benchmarking Suite for Llama 3.2 1B Expert Augmentation
Evaluates throughput, memory usage, and quality metrics
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
import numpy as np
from typing import Optional, Dict
from tqdm import tqdm

from config import ExpertConfig
from surgery import load_augmented_model


class ExpertModelBenchmark:
    """Comprehensive benchmarking suite"""

    def __init__(
        self,
        model_path: str,
        is_augmented: bool = True,
        config: Optional[ExpertConfig] = None,
        use_auth_token: Optional[str] = None
    ):
        """
        Initialize benchmark

        Args:
            model_path: Path to model or HuggingFace model ID
            is_augmented: Whether model has expert augmentation
            config: ExpertConfig for augmented models
            use_auth_token: HuggingFace auth token
        """
        self.model_path = model_path
        self.is_augmented = is_augmented
        self.use_auth_token = use_auth_token

        print(f"Loading model from: {model_path}")

        if is_augmented:
            self.model, self.config = load_augmented_model(
                model_path, config, use_auth_token
            )
        else:
            # Load base model for comparison
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                token=use_auth_token
            )
            self.config = config or ExpertConfig()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=use_auth_token,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()

    def benchmark_throughput(
        self,
        num_samples: int = 100,
        max_new_tokens: int = 50,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """Measure generation throughput in tokens/second"""
        print(f"\nBenchmarking throughput ({num_samples} samples)...")

        # Load test data
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in dataset["text"][:num_samples * 2] if len(t) > 50][:num_samples]

        # Prepare inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        total_tokens = 0
        start_time = time.time()

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
                batch_inputs = {
                    k: v[i:i+batch_size].to(self.model.device)
                    for k, v in inputs.items()
                }

                outputs = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                total_tokens += outputs.shape[0] * max_new_tokens

        end_time = time.time()
        elapsed_time = end_time - start_time
        throughput = total_tokens / elapsed_time

        return {
            "throughput_tokens_per_sec": throughput,
            "total_tokens_generated": total_tokens,
            "elapsed_time_sec": elapsed_time,
        }

    def benchmark_memory(self) -> Dict[str, float]:
        """Measure peak memory usage"""
        print("\nBenchmarking memory usage...")

        if not torch.cuda.is_available():
            print("  Warning: CUDA not available, skipping memory benchmark")
            return {"peak_memory_gb": 0.0}

        torch.cuda.reset_peak_memory_stats()

        # Create dummy input
        dummy_input = torch.randint(
            0, self.tokenizer.vocab_size, (1, 2048)
        ).to(self.model.device)

        with torch.no_grad():
            _ = self.model(dummy_input)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

        return {
            "peak_memory_gb": peak_memory,
            "model_size_gb": sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            ) / 1024**3,
        }

    def benchmark_perplexity(
        self,
        max_length: int = 2048,
        stride: int = 512,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """Calculate perplexity on WikiText-2"""
        print("\nBenchmarking perplexity on WikiText-2...")

        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])

        if max_samples:
            text = text[:max_samples * 1000]  # Approximate character limit

        encodings = self.tokenizer(text, return_tensors="pt")

        seq_len = encodings.input_ids.size(1)
        nlls = []
        prev_end_loc = 0

        pbar = tqdm(range(0, seq_len, stride), desc="Computing perplexity")

        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc

            if end_loc == seq_len:
                break

            # Update progress bar
            pbar.set_postfix({"current_ppl": torch.exp(torch.stack(nlls).sum() / end_loc).item()})

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

        return {
            "perplexity": ppl.item(),
            "tokens_evaluated": end_loc,
        }

    def benchmark_quality_samples(self, num_samples: int = 10) -> Dict[str, any]:
        """Generate sample outputs for qualitative evaluation"""
        print(f"\nGenerating {num_samples} quality samples...")

        prompts = [
            "The key to success is",
            "Artificial intelligence will",
            "In the future, technology",
            "The most important thing in life is",
            "Science has shown that",
            "The history of humanity shows",
            "Climate change requires",
            "The benefits of education include",
            "Innovation happens when",
            "The future of work will",
        ][:num_samples]

        samples = []

        for prompt in tqdm(prompts, desc="Generating samples"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples.append({"prompt": prompt, "generated": generated_text})

        return {"samples": samples}

    def run_full_benchmark(
        self,
        include_perplexity: bool = True,
        include_samples: bool = True
    ) -> Dict[str, any]:
        """Execute complete benchmark suite"""
        results = {}

        # Throughput
        results.update(self.benchmark_throughput())

        # Memory
        results.update(self.benchmark_memory())

        # Perplexity
        if include_perplexity:
            results.update(self.benchmark_perplexity())

        # Quality samples
        if include_samples:
            results.update(self.benchmark_quality_samples())

        return results


def run_full_benchmark(
    model_path: str = "./llama-3.2-1b-expert",
    is_augmented: bool = True,
    config: Optional[ExpertConfig] = None,
    use_auth_token: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, any]:
    """Execute complete benchmark suite and optionally save results"""
    benchmark = ExpertModelBenchmark(
        model_path, is_augmented, config, use_auth_token
    )

    results = benchmark.run_full_benchmark()

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    for key, value in results.items():
        if key != "samples":
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    if "samples" in results:
        print("\n" + "="*60)
        print("QUALITY SAMPLES")
        print("="*60)
        for i, sample in enumerate(results["samples"][:3], 1):
            print(f"\nSample {i}:")
            print(f"Prompt: {sample['prompt']}")
            print(f"Generated: {sample['generated']}\n")

    # Save results if requested
    if output_file:
        import json
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")

    return results


def compare_models(
    base_model_path: str = "meta-llama/Llama-3.2-1B",
    augmented_model_path: str = "./llama-3.2-1b-expert",
    use_auth_token: Optional[str] = None
):
    """Compare base and augmented models"""
    print("\n" + "="*60)
    print("COMPARING BASE VS AUGMENTED MODEL")
    print("="*60)

    # Benchmark base model
    print("\n[1/2] Benchmarking BASE model...")
    base_results = run_full_benchmark(
        base_model_path,
        is_augmented=False,
        use_auth_token=use_auth_token
    )

    # Benchmark augmented model
    print("\n[2/2] Benchmarking AUGMENTED model...")
    augmented_results = run_full_benchmark(
        augmented_model_path,
        is_augmented=True,
        use_auth_token=use_auth_token
    )

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    for key in ["throughput_tokens_per_sec", "peak_memory_gb", "perplexity"]:
        if key in base_results and key in augmented_results:
            base_val = base_results[key]
            aug_val = augmented_results[key]
            improvement = ((aug_val - base_val) / base_val) * 100

            print(f"\n{key}:")
            print(f"  Base: {base_val:.4f}")
            print(f"  Augmented: {aug_val:.4f}")
            print(f"  Change: {improvement:+.2f}%")


if __name__ == "__main__":
    # Test benchmarking
    run_full_benchmark()
