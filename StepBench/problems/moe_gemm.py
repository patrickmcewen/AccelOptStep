import torch
import torch.nn as nn

batch = 64
dim = 2048
n_experts = 8
top_k = 2
expert_dim = 4096


class Model(nn.Module):
    def __init__(self, dim, n_experts, top_k, expert_dim):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Linear(dim, expert_dim, bias=False) for _ in range(n_experts)
        ])

    def forward(self, x):
        # x: (batch, dim)
        scores = torch.softmax(self.gate(x), dim=-1)  # (batch, n_experts)
        topk_weights, topk_indices = scores.topk(self.top_k, dim=-1)  # (batch, top_k)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros(x.size(0), self.experts[0].out_features, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            idx = topk_indices[:, k]  # (batch,)
            w = topk_weights[:, k].unsqueeze(-1)  # (batch, 1)
            for e in range(len(self.experts)):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
        return out


def get_inputs():
    return [torch.randn(batch, dim)]


def get_init_inputs():
    return [dim, n_experts, top_k, expert_dim]
