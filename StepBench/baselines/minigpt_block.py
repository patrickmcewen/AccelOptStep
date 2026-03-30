"""MiniGPT transformer block baseline.

Precomputes the first half (LayerNorm1 + CausalSelfAttention + Residual)
on the host, then expresses the second half in STeP:

  x_ln2  = LayerNorm2(x2)
  fc1    = x_ln2 @ W_fc^T + b_fc                # (B*T, 4C)
  gelu   = GELU(fc1)                             # (B*T, 4C)
  fc2    = gelu  @ W_proj^T + b_proj             # (B*T, C)
  output = fc2 + x2                              # residual add

The attention half is precomputed because multi-head reshape (B*T, C) <->
(B*nh, T, hs) cannot be expressed as a simple stride pattern in STeP.

GELU is composed from STeP primitives:
  GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx import MultiDiGraph

from step_py.functions import map_fn
from step_py.ops import OffChipLoad, UnaryMap, BinaryMap, Broadcast, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42
NR_ITERS = 4


def _load_const(tile_m, M_tiles, val, par_dispatch):
    """Scalar constant broadcast across stream positions. Tile: (tile_m, 1)."""
    t = torch.full((tile_m, 1), val, dtype=torch.float32)
    return OffChipLoad(underlying=t, stride=(0, 0),
                       out_shape_tiled=(M_tiles, 1),
                       tile_row=tile_m, tile_col=1, par_dispatch=par_dispatch)


def _nr_rsqrt_iter(graph, y_prev, s, const_1_5, const_neg_0_5, bw_mul, bw_add):
    """Newton-Raphson iteration: y_new = y * (1.5 + (-0.5)*s*y^2)."""
    y_bc = Broadcast(graph=graph, input=y_prev, num_consumers=3)
    y_sq = BinaryMap(graph=graph, in1=(y_bc, 0), in2=(y_bc, 1),
                     fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    s_y_sq = BinaryMap(graph=graph, in1=s, in2=y_sq,
                       fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    neg_half_s_y_sq = BinaryMap(graph=graph, in1=const_neg_0_5, in2=s_y_sq,
                                fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    bracket = BinaryMap(graph=graph, in1=const_1_5, in2=neg_half_s_y_sq,
                        fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    y_new = BinaryMap(graph=graph, in1=(y_bc, 2), in2=bracket,
                      fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    return y_new


def _build_layernorm(graph, x_stream, weight_1d, bias_1d,
                     tile_m, M_tiles, C, par_dispatch,
                     bw_mul, bw_add, bw_div):
    """LayerNorm subgraph. Input/output: stream (M_tiles, 1) of (tile_m, C)."""
    eps = 1e-5

    x_bc = Broadcast(graph=graph, input=x_stream, num_consumers=2)

    # mean = RowWiseSum(x) / C
    sum_x = UnaryMap(graph=graph, input=(x_bc, 0),
                     fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=bw_div)
    c_load = _load_const(tile_m, M_tiles, float(C), par_dispatch)
    mean = BinaryMap(graph=graph, in1=sum_x, in2=c_load,
                     fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)

    # x_centered = x - mean
    neg_one = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    neg_mean = BinaryMap(graph=graph, in1=mean, in2=neg_one,
                         fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    x_centered = BinaryMap(graph=graph, in1=(x_bc, 1), in2=neg_mean,
                           fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    xc_bc = Broadcast(graph=graph, input=x_centered, num_consumers=3)

    # var = RowWiseSum(x_centered^2) / C
    xc_sq = BinaryMap(graph=graph, in1=(xc_bc, 0), in2=(xc_bc, 1),
                      fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    sum_sq = UnaryMap(graph=graph, input=xc_sq,
                      fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=bw_div)
    c_load2 = _load_const(tile_m, M_tiles, float(C), par_dispatch)
    var = BinaryMap(graph=graph, in1=sum_sq, in2=c_load2,
                    fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)

    # s = var + eps
    eps_ld = _load_const(tile_m, M_tiles, eps, par_dispatch)
    s = BinaryMap(graph=graph, in1=var, in2=eps_ld,
                  fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    # rsqrt(s) via Pade initial estimate + Newton-Raphson refinement
    s_bc = Broadcast(graph=graph, input=s, num_consumers=NR_ITERS + 2)

    neg1 = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    s_m1 = BinaryMap(graph=graph, in1=(s_bc, 0), in2=neg1,
                     fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    one = _load_const(tile_m, M_tiles, 1.0, par_dispatch)
    s_p1 = BinaryMap(graph=graph, in1=(s_bc, 1), in2=one,
                     fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    ratio = BinaryMap(graph=graph, in1=s_m1, in2=s_p1,
                      fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)
    neg1b = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    neg_ratio = BinaryMap(graph=graph, in1=neg1b, in2=ratio,
                          fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    y = UnaryMap(graph=graph, input=neg_ratio,
                 fn=map_fn.Exp(), write_back_mu=False, compute_bw=bw_mul)

    for i in range(NR_ITERS):
        c15 = _load_const(tile_m, M_tiles, 1.5, par_dispatch)
        cn05 = _load_const(tile_m, M_tiles, -0.5, par_dispatch)
        y = _nr_rsqrt_iter(graph, y, (s_bc, i + 2), c15, cn05, bw_mul, bw_add)

    # normalized = x_centered * rsqrt
    normalized = BinaryMap(graph=graph, in1=(xc_bc, 2), in2=y,
                           fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

    # scale and shift: output = normalized * weight + bias
    w_2d = weight_1d.unsqueeze(0).expand(tile_m, C).contiguous()
    w_ld = OffChipLoad(underlying=w_2d, stride=(0, 0),
                       out_shape_tiled=(M_tiles, 1),
                       tile_row=tile_m, tile_col=C, par_dispatch=par_dispatch)
    scaled = BinaryMap(graph=graph, in1=normalized, in2=w_ld,
                       fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

    b_2d = bias_1d.unsqueeze(0).expand(tile_m, C).contiguous()
    b_ld = OffChipLoad(underlying=b_2d, stride=(0, 0),
                       out_shape_tiled=(M_tiles, 1),
                       tile_row=tile_m, tile_col=C, par_dispatch=par_dispatch)
    result = BinaryMap(graph=graph, in1=scaled, in2=b_ld,
                       fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    return result


def _build_gelu(graph, x_stream, tile_m, M_tiles, par_dispatch,
                bw_mul, bw_add, bw_div):
    """GELU subgraph composed from STeP primitives.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)

    # x used 5 times: (0,1) -> x^2, (2) -> x^3, (3) -> inner, (4) -> final
    x_bc = Broadcast(graph=graph, input=x_stream, num_consumers=5)

    # x^2
    x_sq = BinaryMap(graph=graph, in1=(x_bc, 0), in2=(x_bc, 1),
                     fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    # x^3
    x_cube = BinaryMap(graph=graph, in1=x_sq, in2=(x_bc, 2),
                       fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    # 0.044715 * x^3
    c_coeff = _load_const(tile_m, M_tiles, 0.044715, par_dispatch)
    coeff_x3 = BinaryMap(graph=graph, in1=c_coeff, in2=x_cube,
                         fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    # inner = x + 0.044715 * x^3
    inner = BinaryMap(graph=graph, in1=(x_bc, 3), in2=coeff_x3,
                      fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    # z = sqrt(2/pi) * inner
    c_s2p = _load_const(tile_m, M_tiles, sqrt_2_over_pi, par_dispatch)
    z = BinaryMap(graph=graph, in1=c_s2p, in2=inner,
                  fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    # 2z
    c_2 = _load_const(tile_m, M_tiles, 2.0, par_dispatch)
    two_z = BinaryMap(graph=graph, in1=c_2, in2=z,
                      fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    # exp(2z)
    exp_2z = UnaryMap(graph=graph, input=two_z,
                      fn=map_fn.Exp(), write_back_mu=False, compute_bw=bw_mul)

    # tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    exp_bc = Broadcast(graph=graph, input=exp_2z, num_consumers=2)
    c_neg1 = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    num = BinaryMap(graph=graph, in1=(exp_bc, 0), in2=c_neg1,
                    fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    c_1 = _load_const(tile_m, M_tiles, 1.0, par_dispatch)
    den = BinaryMap(graph=graph, in1=(exp_bc, 1), in2=c_1,
                    fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    tanh_z = BinaryMap(graph=graph, in1=num, in2=den,
                       fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)

    # 1 + tanh(z)
    c_1b = _load_const(tile_m, M_tiles, 1.0, par_dispatch)
    one_plus_tanh = BinaryMap(graph=graph, in1=c_1b, in2=tanh_z,
                              fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    # 0.5 * x
    c_05 = _load_const(tile_m, M_tiles, 0.5, par_dispatch)
    half_x = BinaryMap(graph=graph, in1=c_05, in2=(x_bc, 4),
                       fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    # result = 0.5 * x * (1 + tanh(z))
    result = BinaryMap(graph=graph, in1=half_x, in2=one_plus_tanh,
                       fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

    return result


def build_graph(dims):
    B = dims["batch_size"]
    T = dims["seq_len"]
    C = dims["n_embd"]
    n_head = dims["n_head"]
    max_seqlen = dims["max_seqlen"]
    four_C = 4 * C
    M = B * T

    tile_m = min(64, M)
    assert M % tile_m == 0
    M_tiles = M // tile_m
    par_dispatch = 4

    bw_mul = 128
    bw_add = 128
    bw_div = 128
    bw_matmul = 512

    graph = MultiDiGraph()

    # ===== Replicate model weights (same RNG sequence as compute_gold) =====
    torch.manual_seed(SEED)
    ln_1 = nn.LayerNorm(C)
    c_attn = nn.Linear(C, 3 * C)
    c_proj_attn = nn.Linear(C, C)
    ln_2 = nn.LayerNorm(C)
    c_fc = nn.Linear(C, four_C)
    c_proj_mlp = nn.Linear(four_C, C)

    torch.manual_seed(SEED)
    x_3d = torch.randn(B, T, C, dtype=torch.float32)

    # ===== First half on host: LN1 -> Attention -> Residual =====
    with torch.no_grad():
        x_ln1 = F.layer_norm(x_3d, (C,), ln_1.weight, ln_1.bias)

        qkv = F.linear(x_ln1, c_attn.weight, c_attn.bias)
        q, k, v = qkv.split(C, dim=2)

        hs = C // n_head
        q = q.view(B, T, n_head, hs).transpose(1, 2)
        k = k.view(B, T, n_head, hs).transpose(1, 2)
        v = v.view(B, T, n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        causal = torch.tril(torch.ones(max_seqlen, max_seqlen))
        att = att.masked_fill(causal[:T, :T].view(1, 1, T, T) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y_attn = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y_attn = F.linear(y_attn, c_proj_attn.weight, c_proj_attn.bias)

        x2 = x_3d + y_attn

    # ===== Second half in STeP: LN2 -> MLP(FC1 -> GELU -> FC2) -> Residual =====
    x2_2d = x2.reshape(M, C).contiguous()

    # --- Load x2, broadcast to (residual, LN2) ---
    x2_load = OffChipLoad(
        underlying=x2_2d,
        stride=(1, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=tile_m, tile_col=C,
        par_dispatch=par_dispatch,
    )
    x2_bc = Broadcast(graph=graph, input=x2_load, num_consumers=2)

    # --- LayerNorm2 ---
    x_ln2 = _build_layernorm(
        graph, (x2_bc, 0), ln_2.weight.detach(), ln_2.bias.detach(),
        tile_m, M_tiles, C, par_dispatch,
        bw_mul, bw_add, bw_div,
    )

    # --- FC1: (tile_m, C) @ (4C, C)^T -> (tile_m, 4C) ---
    wfc_load = OffChipLoad(
        underlying=c_fc.weight.detach().contiguous(),
        stride=(0, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=four_C, tile_col=C,
        par_dispatch=par_dispatch,
    )
    fc1 = BinaryMap(
        graph=graph, in1=x_ln2, in2=wfc_load,
        fn=map_fn.Matmul(weight_transposed=True),
        write_back_mu=False, compute_bw=bw_matmul,
    )

    # FC1 bias: (1, 4C) broadcasts to (tile_m, 4C)
    bfc_load = OffChipLoad(
        underlying=c_fc.bias.detach().unsqueeze(0).contiguous(),
        stride=(0, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=1, tile_col=four_C,
        par_dispatch=par_dispatch,
    )
    fc1_biased = BinaryMap(
        graph=graph, in1=fc1, in2=bfc_load,
        fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add,
    )

    # --- GELU ---
    gelu_out = _build_gelu(
        graph, fc1_biased, tile_m, M_tiles, par_dispatch,
        bw_mul, bw_add, bw_div,
    )

    # --- FC2: (tile_m, 4C) @ (C, 4C)^T -> (tile_m, C) ---
    wproj_load = OffChipLoad(
        underlying=c_proj_mlp.weight.detach().contiguous(),
        stride=(0, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=C, tile_col=four_C,
        par_dispatch=par_dispatch,
    )
    fc2 = BinaryMap(
        graph=graph, in1=gelu_out, in2=wproj_load,
        fn=map_fn.Matmul(weight_transposed=True),
        write_back_mu=False, compute_bw=bw_matmul,
    )

    # FC2 bias: (1, C) broadcasts to (tile_m, C)
    bproj_load = OffChipLoad(
        underlying=c_proj_mlp.bias.detach().unsqueeze(0).contiguous(),
        stride=(0, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=1, tile_col=C,
        par_dispatch=par_dispatch,
    )
    fc2_biased = BinaryMap(
        graph=graph, in1=fc2, in2=bproj_load,
        fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add,
    )

    # --- Residual: output = fc2 + x2 ---
    output = BinaryMap(
        graph=graph, in1=fc2_biased, in2=(x2_bc, 1),
        fn=map_fn.Add(), write_back_mu=True, compute_bw=bw_add,
    )

    output_op = OffChipStore(
        graph=graph, input=output,
        par_dispatch=par_dispatch, store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
