"""
Core Insertable Modules for Llama 3.2 1B Expert Augmentation
Implements MoE adapters, routing mechanisms, and efficiency modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class NoisyTopKGating(nn.Module):
    """Stabilized routing with auxiliary losses"""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, config):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.config = config

        self.w_gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.w_noise = nn.Linear(hidden_size, num_experts, bias=False)

        self.softplus = nn.Softplus()
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))

    def _gates_to_load(self, gates):
        """Compute load per expert"""
        return (gates > 0).float().sum(0)

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_logits = self.w_gate(x)

        if training:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-10
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # Create sparse gate matrix
        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_k_indices, top_k_gates)

        # Load balancing loss
        loads = self._gates_to_load(gates)
        importance = gates.sum(0)
        load_balancing_loss = self.config.router_aux_loss_coef * (
            self.num_experts * torch.sum(importance * loads) / (gates.shape[0] ** 2)
        )

        # Router z-loss for stability
        router_z_loss = self.config.router_z_loss_coef * torch.mean(
            torch.logsumexp(clean_logits, dim=-1) ** 2
        )

        return gates, load_balancing_loss + router_z_loss


class ExpertFFN(nn.Module):
    """Enhanced expert FFN with LoRA integration"""

    def __init__(self, hidden_size: int, intermediate_size: int, config):
        super().__init__()
        self.config = config

        # Gate projection with LoRA
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_lora_A = nn.Linear(hidden_size, config.lora_r, bias=False)
        self.gate_lora_B = nn.Linear(config.lora_r, intermediate_size, bias=False)

        # Up projection with LoRA
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_lora_A = nn.Linear(hidden_size, config.lora_r, bias=False)
        self.up_lora_B = nn.Linear(config.lora_r, intermediate_size, bias=False)

        # Down projection with LoRA
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.down_lora_A = nn.Linear(intermediate_size, config.lora_r, bias=False)
        self.down_lora_B = nn.Linear(config.lora_r, hidden_size, bias=False)

        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.expert_dropout)

        # Initialize LoRA layers
        self._init_lora_weights()

    def _init_lora_weights(self):
        """Initialize LoRA weights using Kaiming initialization"""
        nn.init.kaiming_uniform_(self.gate_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_lora_B.weight)

        nn.init.kaiming_uniform_(self.up_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_lora_B.weight)

        nn.init.kaiming_uniform_(self.down_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA-enhanced projections with scaling
        scaling = self.config.lora_alpha / self.config.lora_r

        gate = self.gate_proj(x) + scaling * self.gate_lora_B(self.gate_lora_A(x))
        up = self.up_proj(x) + scaling * self.up_lora_B(self.up_lora_A(x))

        # Activation
        activated = self.act_fn(gate) * up
        activated = self.dropout(activated)

        # Down projection
        down = self.down_proj(activated) + scaling * self.down_lora_B(self.down_lora_A(activated))

        return down


class MoEAdapter(nn.Module):
    """Insertable MoE adapter with layer-wise capacity"""

    def __init__(self, hidden_size: int, layer_idx: int, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Layer-wise expert allocation
        num_experts = config.layer_wise_expert_allocation.get(layer_idx, config.num_experts)

        self.gating = NoisyTopKGating(hidden_size, num_experts, config.top_k, config)
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_size, config.expert_hidden_size, config)
            for _ in range(num_experts)
        ])

        # Residual scaling
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_size)

        # Routing
        gates, routing_loss = self.gating(x, training)

        # Expert computation
        expert_outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_mask = gates[:, i] > 0
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_output = expert(expert_input)
                expert_outputs[expert_mask] += expert_output * gates[expert_mask, i][:, None]

        # Residual connection with scaling
        output = hidden_states + self.alpha * expert_outputs.reshape(original_shape)

        return output, routing_loss


class KVCacheCompressor(nn.Module):
    """Insertable KV-cache compression for efficiency"""

    def __init__(self, num_heads: int, head_dim: int, compression_ratio: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.compressed_heads = max(1, int(num_heads * compression_ratio))

        # Compression projections
        self.k_compress = nn.Linear(
            num_heads * head_dim, self.compressed_heads * head_dim, bias=False
        )
        self.v_compress = nn.Linear(
            num_heads * head_dim, self.compressed_heads * head_dim, bias=False
        )
        self.k_decompress = nn.Linear(
            self.compressed_heads * head_dim, num_heads * head_dim, bias=False
        )
        self.v_decompress = nn.Linear(
            self.compressed_heads * head_dim, num_heads * head_dim, bias=False
        )

    def forward(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = key_states.shape

        # Compress
        compressed_k = self.k_compress(key_states)
        compressed_v = self.v_compress(value_states)

        # Decompress for compatibility
        decompressed_k = self.k_decompress(compressed_k)
        decompressed_v = self.v_decompress(compressed_v)

        return decompressed_k, decompressed_v
