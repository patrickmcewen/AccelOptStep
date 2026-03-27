# Origin: /home/ubuntu/patrick/AbstractOpt/KernelBench/KernelBench/level1/97_ScaledDotProductAttention.py
# Notes: Direct copy
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out

SEED = 42

def get_inputs(dims):
    torch.manual_seed(SEED)
    batch, heads, seq, dim = dims["batch"], dims["heads"], dims["seq"], dims["dim"]
    Q = torch.randn(batch, heads, seq, dim)
    K = torch.randn(batch, heads, seq, dim)
    V = torch.randn(batch, heads, seq, dim)
    return [Q, K, V]

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
