import torch
import torch.nn as nn

batch = 4
seq_len = 2048
d_model = 4096
d_ff = 11008
eps = 1e-6


class Model(nn.Module):
    def __init__(self, d_model, d_ff, eps):
        super().__init__()
        self.eps = eps
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # RMSNorm
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x * rms
        # SwiGLU FFN
        return self.w_down(torch.nn.functional.silu(self.w_gate(normed)) * self.w_up(normed))


def get_inputs():
    return [torch.randn(batch, seq_len, d_model)]


def get_init_inputs():
    return [d_model, d_ff, eps]
