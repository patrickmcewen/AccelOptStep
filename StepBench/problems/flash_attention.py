import torch
import torch.nn as nn
import math

batch = 4
heads = 32
seq = 2048
head_dim = 128


class Model(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, Q, K, V):
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, V)


def get_inputs():
    return [
        torch.randn(batch, heads, seq, head_dim),
        torch.randn(batch, heads, seq, head_dim),
        torch.randn(batch, heads, seq, head_dim),
    ]


def get_init_inputs():
    return [head_dim]
