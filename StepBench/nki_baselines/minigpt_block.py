"""NKI baseline for minigpt_block: full transformer block.

Operations:
  1. LayerNorm -> CausalSelfAttention -> residual add
  2. LayerNorm -> MLP (Linear -> GELU -> Linear) -> residual add

Uses run_nki(dims) to chain sub-operations in numpy, since the full
transformer block is too complex for a single @nki.jit kernel.
Each sub-operation mirrors the PyTorch reference exactly.
"""

import math
import numpy as np
import torch
import torch.nn as nn

SEED = 42
EPS = 1e-5


def _layernorm(x, weight, bias):
    """LayerNorm over last dimension. x: (..., C), weight/bias: (C,)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * weight + bias


def _linear(x, weight, bias):
    """nn.Linear: x @ weight^T + bias. weight: (out, in), bias: (out,)."""
    return x @ weight.T + bias


def _new_gelu(x):
    """NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3.0))))


def _softmax(x, axis=-1):
    """Numerically stable softmax."""
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _causal_self_attention(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, n_head):
    """CausalSelfAttention forward pass.

    x: (B, T, C)
    c_attn_w: (3*C, C), c_attn_b: (3*C,)
    c_proj_w: (C, C), c_proj_b: (C,)
    """
    B, T, C = x.shape
    head_dim = C // n_head

    # QKV projection
    qkv = _linear(x, c_attn_w, c_attn_b)  # (B, T, 3*C)
    q, k, v = np.split(qkv, 3, axis=2)     # each (B, T, C)

    # Reshape to (B, n_head, T, head_dim)
    q = q.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)

    # Attention scores: (B, n_head, T, T)
    scale = 1.0 / math.sqrt(head_dim)
    att = (q @ k.transpose(0, 1, 3, 2)) * scale

    # Causal mask: upper triangle gets -inf
    causal_mask = np.triu(np.ones((T, T), dtype=np.float32), k=1)
    att = np.where(causal_mask == 1, -np.inf, att)

    # Softmax
    att = _softmax(att, axis=-1)

    # Weighted sum: (B, n_head, T, head_dim)
    y = att @ v

    # Reshape back: (B, T, C)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

    # Output projection
    y = _linear(y, c_proj_w, c_proj_b)
    return y


def run_nki(dims):
    """Execute the full transformer block using numpy sub-operations.

    Extracts weights from the PyTorch model (seeded identically), then
    runs the forward pass in numpy to produce the output.
    """
    from StepBench.problems.minigpt_block import Model, get_init_inputs

    n_embd = dims["n_embd"]
    n_head = dims["n_head"]

    # Build the PyTorch model to extract weights
    torch.manual_seed(SEED)
    model = Model(*get_init_inputs(dims))
    params = {name: param.detach().numpy() for name, param in model.named_parameters()}

    # Generate input (same seed as get_inputs)
    torch.manual_seed(SEED)
    x = torch.randn(dims["batch_size"], dims["seq_len"], n_embd).numpy()

    # --- Block forward pass ---

    # 1. LayerNorm 1
    ln1_out = _layernorm(x, params["ln_1.weight"], params["ln_1.bias"])

    # 2. Causal self-attention
    attn_out = _causal_self_attention(
        ln1_out,
        params["attn.c_attn.weight"], params["attn.c_attn.bias"],
        params["attn.c_proj.weight"], params["attn.c_proj.bias"],
        n_head,
    )

    # 3. Residual connection
    x = x + attn_out

    # 4. LayerNorm 2
    ln2_out = _layernorm(x, params["ln_2.weight"], params["ln_2.bias"])

    # 5. MLP: Linear -> GELU -> Linear
    mlp_hidden = _linear(ln2_out, params["mlp.c_fc.weight"], params["mlp.c_fc.bias"])
    mlp_act = _new_gelu(mlp_hidden)
    mlp_out = _linear(mlp_act, params["mlp.c_proj.weight"], params["mlp.c_proj.bias"])

    # 6. Residual connection
    x = x + mlp_out

    return x
