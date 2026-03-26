import torch
import torch.nn as nn
import math

batch = 4
seq_len = 2048
d_model = 4096
n_heads = 32
d_ff = 11008
eps = 1e-6


class Model(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, eps):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps

        # Attention projections
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # FFN projections
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def _rmsnorm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        b, l, d = x.shape

        # Pre-norm attention
        h = self._rmsnorm(x)
        q = self.wq(h).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, d)
        x = x + self.wo(attn_out)

        # Pre-norm SwiGLU FFN
        h = self._rmsnorm(x)
        x = x + self.w_down(torch.nn.functional.silu(self.w_gate(h)) * self.w_up(h))

        return x


def get_inputs():
    return [torch.randn(batch, seq_len, d_model)]


def get_init_inputs():
    return [d_model, n_heads, d_ff, eps]
