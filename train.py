"""
Production Training Pipeline for Llama 3.2 1B Expert Augmentation
Includes routing loss integration, gradient accumulation, and monitoring
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import wandb
import os
from typing import Optional, Dict, Any

from config import ExpertConfig
from surgery import load_augmented_model, collect_routing_losses


class ExpertAugmentedTrainer(Trainer):
    """Custom trainer that handles routing auxiliary losses"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss including routing auxiliary losses"""
        # Standard language modeling loss
        outputs = model(**inputs)
        loss = outputs.loss

        # Add routing losses from MoE adapters
        routing_loss = collect_routing_losses(model)
        total_loss = loss + routing_loss

        # Log losses
        if self.state.global_step % self.args.logging_steps == 0:
            try:
                wandb.log({
                    "train/lm_loss": loss.item(),
                    "train/routing_loss": routing_loss.item(),
                    "train/total_loss": total_loss.item(),
                })
            except:
                pass  # wandb might not be initialized

        return (total_loss, outputs) if return_outputs else total_loss


def prepare_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    num_samples: Optional[int] = None
):
    """Prepare and tokenize dataset"""
    print(f"Loading dataset: {dataset_name}")

    # Load dataset
    if dataset_name == "HuggingFaceH4/ultrachat_200k":
        dataset = load_dataset(dataset_name, split="train_sft")
    else:
        dataset = load_dataset(dataset_name, split="train")

    # Limit samples if specified
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)} samples")

    # Tokenization function
    def tokenize_function(examples):
        # Handle different dataset formats
        if "messages" in examples:
            # Chat format
            texts = []
            for messages in examples["messages"]:
                if isinstance(messages, list):
                    # Format as conversation
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                else:
                    text = str(messages)
                texts.append(text)
        elif "text" in examples:
            texts = examples["text"]
        else:
            # Fallback: use first text field
            key = next(k for k in examples.keys() if isinstance(examples[k][0], str))
            texts = examples[key]

        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    return tokenized_dataset


def fine_tune_augmented_model(
    model_id: str = "meta-llama/Llama-3.2-1B",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    output_dir: str = "./llama-3.2-1b-expert",
    config: Optional[ExpertConfig] = None,
    num_train_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 100,
    max_samples: Optional[int] = None,
    use_wandb: bool = False,
    wandb_project: str = "llama-expert-augmentation",
    use_auth_token: Optional[str] = None,
):
    """Production fine-tuning with all optimizations"""
    if config is None:
        config = ExpertConfig()

    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project=wandb_project,
            config={
                "model_id": model_id,
                "dataset": dataset_name,
                "num_experts": config.num_experts,
                "top_k": config.top_k,
                "lora_r": config.lora_r,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            }
        )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, expert_config = load_augmented_model(model_id, config, use_auth_token)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=use_auth_token,
        trust_remote_code=True
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare dataset
    if dataset_config:
        full_dataset_name = f"{dataset_name}"
        dataset = load_dataset(dataset_name, dataset_config, split="train")
    else:
        full_dataset_name = dataset_name
        dataset = prepare_dataset(dataset_name, tokenizer, max_length=2048, num_samples=max_samples)

    # For simple datasets like wikitext, do manual tokenization
    if dataset_name.startswith("wikitext"):
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
                padding=False,
            )

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=True,
        optim="adamw_torch",
        report_to="wandb" if use_wandb else "none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=config.use_gradient_checkpointing,
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = ExpertAugmentedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    trainer.train()

    # Save final model
    print("\n" + "="*60)
    print("Saving model...")
    print("="*60 + "\n")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if use_wandb:
        wandb.finish()

    print(f"✓ Training complete! Model saved to {output_dir}")

    return model


if __name__ == "__main__":
    # Test training pipeline
    fine_tune_augmented_model(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        num_train_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=4,
        max_samples=100,
        use_wandb=False,
    )
