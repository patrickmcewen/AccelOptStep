import torch
import torch.nn as nn

M = 2048
K = 4096
N = 4096


class Model(nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.linear = nn.Linear(K, N, bias=False)

    def forward(self, x):
        return torch.nn.functional.silu(self.linear(x))


def get_inputs():
    return [torch.randn(M, K)]


def get_init_inputs():
    return [K, N]
