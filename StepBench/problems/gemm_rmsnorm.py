import torch
import torch.nn as nn

M = 2048
K = 4096
N = 4096
eps = 1e-6


class Model(nn.Module):
    def __init__(self, K, N, eps):
        super().__init__()
        self.linear = nn.Linear(K, N, bias=False)
        self.eps = eps

    def forward(self, x):
        y = self.linear(x)
        rms = torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y * rms


def get_inputs():
    return [torch.randn(M, K)]


def get_init_inputs():
    return [K, N, eps]
