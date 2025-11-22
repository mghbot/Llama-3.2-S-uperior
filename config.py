"""
Meta-Architecture Configuration for Llama 3.2 1B Expert Augmentation
Defines all hyperparameters for MoE adapters, LoRA integration, and optimization
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ExpertConfig:
    """Meta-expert configuration with cross-domain optimizations"""

    # MoE Adapter Configuration
    num_experts: int = 8
    top_k: int = 2
    expert_hidden_size: int = 2048  # 2x model hidden size for capacity
    router_aux_loss_coef: float = 0.01  # Load balancing
    router_z_loss_coef: float = 0.001   # Router logit stabilization

    # LoRA-MoE Hybrid Configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # KV-Cache Compression
    cache_compression_ratio: float = 0.5
    cache_attention_heads: int = 4

    # Training Dynamics
    layer_wise_expert_allocation: Dict[int, int] = field(default_factory=lambda: {
        0: 4, 1: 4, 2: 6, 3: 6, 4: 8, 5: 8, 6: 8, 7: 8,
        8: 6, 9: 6, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4
    })  # Progressive capacity reduction
    router_temperature_annealing: bool = True
    expert_dropout: float = 0.1

    # Efficiency
    use_flash_attention: bool = False  # Set to True if flash-attn is installed
    use_gradient_checkpointing: bool = True
    quantization_4bit: bool = False  # Set to True for 4-bit quantization

    # Training hyperparameters
    training_step: int = 0  # Current training step for temperature annealing

    def update_training_step(self, step: int):
        """Update training step for temperature annealing"""
        self.training_step = step
