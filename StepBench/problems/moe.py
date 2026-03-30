# Simplified Mixture of Experts (MoE) layer following the STeP paper (Section 3.3).
# Top-1 routing: each token is routed to exactly one expert based on a gating network.
# Each expert is a single linear layer (no bias, no SwiGLU).
# Output is placed back in original token order.
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    def __init__(self, num_experts, hidden_dim, intermediate_dim):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, intermediate_dim, bias=False)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (num_tokens, hidden_dim)
        gate_scores = self.gate(x)                  # (B, E)
        expert_idx = gate_scores.argmax(dim=-1)      # (B,) top-1 routing
        B = x.shape[0]
        output = torch.zeros(B, self.experts[0].out_features,
                             dtype=x.dtype, device=x.device)
        for i in range(self.num_experts):
            mask = expert_idx == i
            if mask.any():
                output[mask] = self.experts[i](x[mask])
        return output


def get_inputs(dims):
    torch.manual_seed(SEED)
    return [torch.randn(dims["num_tokens"], dims["hidden_dim"])]


def get_init_inputs(dims):
    return [dims["num_experts"], dims["hidden_dim"], dims["intermediate_dim"]]


def compute_gold(dims):
    torch.manual_seed(SEED)
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    with torch.no_grad():
        return model(*inputs)
