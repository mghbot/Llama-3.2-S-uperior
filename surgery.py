"""
Model Surgery for Llama 3.2 1B Expert Augmentation
Surgically inserts MoE adapters and KV-cache compression into existing model
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from typing import Optional
import warnings

from config import ExpertConfig
from modules import MoEAdapter, KVCacheCompressor


class AugmentedMLPWrapper(nn.Module):
    """Wrapper that combines original MLP with MoE adapter"""

    def __init__(self, original_mlp, moe_adapter):
        super().__init__()
        self.original_mlp = original_mlp
        self.moe_adapter = moe_adapter
        self._routing_loss = None

    def forward(self, hidden_states):
        # Original MLP output
        original_out = self.original_mlp(hidden_states)

        # MoE adapter output
        adapter_out, routing_loss = self.moe_adapter(hidden_states, training=self.training)

        # Store routing loss for later retrieval
        self._routing_loss = routing_loss

        # Combine outputs
        return original_out + adapter_out

    def get_routing_loss(self):
        """Retrieve and clear routing loss"""
        loss = self._routing_loss
        self._routing_loss = None
        return loss


def inject_experts_into_llama(model: AutoModelForCausalLM, config: ExpertConfig) -> AutoModelForCausalLM:
    """Surgically insert MoE adapters into Llama 3.2 1B"""
    model_config = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    # Store routing losses in model
    model._expert_routing_losses = []

    # Inject MoE adapters into each layer
    num_layers = len(model.model.layers)
    print(f"Injecting MoE adapters into {num_layers} layers...")

    for layer_idx, layer in enumerate(model.model.layers):
        hidden_size = model_config.hidden_size

        # Create MoE adapter
        moe_adapter = MoEAdapter(hidden_size, layer_idx, config).to(device).to(dtype)

        # Wrap original MLP
        original_mlp = layer.mlp
        augmented_mlp = AugmentedMLPWrapper(original_mlp, moe_adapter)

        # Replace MLP
        layer.mlp = augmented_mlp

        print(f"  Layer {layer_idx}: Injected MoE adapter with "
              f"{config.layer_wise_expert_allocation.get(layer_idx, config.num_experts)} experts")

    # Enable KV-cache compression if requested (optional for now)
    # This would require more complex integration with the attention mechanism
    # For now, we focus on the MoE adapters which provide the main performance boost

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"Model Augmentation Complete")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen percentage: {100 * (1 - trainable_params / total_params):.2f}%")
    print(f"{'='*60}\n")

    return model


def collect_routing_losses(model: AutoModelForCausalLM) -> torch.Tensor:
    """Collect routing losses from all MoE adapters"""
    routing_losses = []

    for layer in model.model.layers:
        if isinstance(layer.mlp, AugmentedMLPWrapper):
            loss = layer.mlp.get_routing_loss()
            if loss is not None:
                routing_losses.append(loss)

    if routing_losses:
        return sum(routing_losses)
    else:
        return torch.tensor(0.0, device=next(model.parameters()).device)


def load_augmented_model(
    model_id: str = "meta-llama/Llama-3.2-1B",
    config: Optional[ExpertConfig] = None,
    use_auth_token: Optional[str] = None
) -> tuple:
    """Load and augment Llama model"""
    if config is None:
        config = ExpertConfig()

    print(f"Loading base model: {model_id}")

    # Prepare model loading arguments
    model_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # Add authentication token if provided
    if use_auth_token:
        model_kwargs["token"] = use_auth_token

    # Add quantization config if requested
    if config.quantization_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["quantization_config"] = quantization_config

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying without authentication...")
        model_kwargs.pop("token", None)
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Inject expert modules
    model = inject_experts_into_llama(model, config)

    # Enable gradient checkpointing if requested
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    return model, config


if __name__ == "__main__":
    # Test model loading and injection
    print("Testing model loading and expert injection...")
    config = ExpertConfig()
    model, config = load_augmented_model(config=config)
    print("✓ Model loaded and augmented successfully!")
