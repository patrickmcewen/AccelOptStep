import torch
import torch.nn as nn

batch = 64
dim = 2048
n_experts = 8
top_k = 2
inter_dim = 4096


class Expert(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w_gate = nn.Linear(dim, inter_dim, bias=False)
        self.w_up = nn.Linear(dim, inter_dim, bias=False)
        self.w_down = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))


class Model(nn.Module):
    def __init__(self, dim, n_experts, top_k, inter_dim):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(dim, inter_dim) for _ in range(n_experts)
        ])

    def forward(self, x):
        # x: (batch, dim)
        scores = torch.softmax(self.gate(x), dim=-1)  # (batch, n_experts)
        topk_weights, topk_indices = scores.topk(self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk_indices[:, k]
            w = topk_weights[:, k].unsqueeze(-1)
            for e in range(len(self.experts)):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
        return out


def get_inputs():
    return [torch.randn(batch, dim)]


def get_init_inputs():
    return [dim, n_experts, top_k, inter_dim]
