"""
Mixture of Experts (MoE) Layer.

Architecture:
    - 128 experts per layer, top-2 routing
    - Each expert is a SwiGLU FFN (gate, up, down projections)
    - Router: single linear layer [hidden_size → num_experts]
    - Auxiliary load-balancing loss prevents expert collapse

Parameter count:
    Router gate  : 128 × 16384  ≈   2M  params
    Per expert   : 3 × 65536 × 16384 ≈ 3.22B params
    128 experts  : ~412B params per layer
    160 layers   : ~65.9T total MoE params
    Active/token : 2 experts × 3.22B ≈ 6.4B active FFN params

References:
    - Switch Transformer: Fedus et al. 2021 (arXiv:2101.03961)
    - GLaM: Du et al. 2021
    - Mixtral 8×7B: Jiang et al. 2024 (arXiv:2401.04088)
    - DeepSeekMoE: Dai et al. 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """
    Single FFN expert using SwiGLU activation.

    SwiGLU(x) = W_down( SiLU(W_gate · x) ⊙ (W_up · x) )

    Args:
        hidden_size       : 16384
        intermediate_size : 65536
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, hidden_size]
        Returns:
            [num_tokens, hidden_size]
        """
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer with top-k token routing.

    Forward pass:
        1. Router computes logits for each token → [B*T, num_experts]
        2. Softmax + top-k selection → routing weights + expert indices
        3. Tokens dispatched to selected experts
        4. Expert outputs weighted-summed back
        5. Load-balancing auxiliary loss computed

    Auxiliary loss (Fedus et al.):
        L_aux = num_experts × Σ_i (f_i × P_i)
        f_i = fraction of tokens routed to expert i
        P_i = mean routing probability for expert i
        Minimizing this encourages uniform expert utilization.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts  = config.num_experts           # 128
        self.top_k        = config.num_experts_per_tok   # 2
        self.hidden_size  = config.hidden_size           # 16384
        self.aux_loss_coef = getattr(
            config, "router_aux_loss_coef", 0.02
        )

        # Router: linear layer maps hidden → expert logits
        self.gate = nn.Linear(
            self.hidden_size, self.num_experts, bias=False
        )

        # All expert FFNs
        self.experts = nn.ModuleList([
            Expert(config.hidden_size, config.intermediate_size)
            for _ in range(self.num_experts)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, hidden_size]
        Returns:
            output:   [B, T, hidden_size]
            aux_loss: scalar tensor — load balancing loss
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)           # [N, D]  where N = B*T
        N = x_flat.shape[0]

        # --- Routing ---
        router_logits = self.gate(x_flat)    # [N, num_experts]

        # Softmax probabilities used for weighted sum + aux loss
        routing_weights = F.softmax(
            router_logits, dim=-1, dtype=torch.float32
        )  # [N, num_experts]

        # Top-k selection
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # both [N, top_k]

        # Renormalize selected weights to sum to 1
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True
        )  # [N, top_k]

        # Cast back to model dtype for mixing
        topk_weights = topk_weights.to(x_flat.dtype)

        # --- Dispatch & Aggregate ---
        final_output = torch.zeros_like(x_flat)  # [N, D]

        # Efficient batched dispatch
        flat_expert_indices = topk_indices.view(-1)  # [N * top_k]
        flat_token_indices  = (
            torch.arange(N, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )  # [N * top_k]
        flat_weights = topk_weights.view(-1)         # [N * top_k]

        # Group by expert for efficient batching
        for expert_idx in range(self.num_experts):
            mask = flat_expert_indices == expert_idx  # [N * top_k]
            if not mask.any():
                continue

            token_ids  = flat_token_indices[mask]
            weights    = flat_weights[mask]

            expert_input  = x_flat[token_ids]
            expert_output = self.experts[expert_idx](expert_input)

            # Weighted accumulation
            final_output.index_add_(
                0, token_ids,
                expert_output * weights.unsqueeze(-1)
            )

        # --- Auxiliary Load-Balancing Loss ---
        aux_loss = self._load_balance_loss(routing_weights, topk_indices, N)

        return final_output.view(B, T, D), aux_loss

    def _load_balance_loss(
        self,
        routing_weights: torch.Tensor,  # [N, num_experts]  float32
        selected: torch.Tensor,         # [N, top_k]
        num_tokens: int,
    ) -> torch.Tensor:
        """
        Switch Transformer auxiliary loss.

        L_aux = num_experts × Σ_i (f_i × P_i)

        f_i: fraction of tokens dispatched to expert i
        P_i: mean soft routing probability for expert i
        """
        one_hot = F.one_hot(
            selected, num_classes=self.num_experts
        ).float()                              # [N, top_k, num_experts]
        f = one_hot.sum(dim=1).sum(dim=0) / (num_tokens * self.top_k)

        p = routing_weights.mean(dim=0)

        aux_loss = self.num_experts * (f * p).sum()
        return self.aux_loss_coef * aux_loss

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"hidden={self.hidden_size}, "
            f"aux_loss_coef={self.aux_loss_coef}"
        )
