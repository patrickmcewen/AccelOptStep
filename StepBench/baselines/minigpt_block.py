"""MiniGPT transformer block — full STeP baseline.

Expresses the complete transformer block in STeP IR:
  LN1 → Per-Head QKV Projections → Causal Attention → Output Projection
  → Residual → LN2 → FC1 → GELU → FC2 → Residual

Key techniques:
  - Per-head computation avoids the multi-head transpose entirely.
    Each head h gets its own W_q_h, W_k_h, W_v_h slices of shape (hs, C).
  - Output projection decomposed as sum of per-head matmuls:
    y = sum_h(attn_h @ W_proj_h^T) + bias
  - Causal mask applied as additive -inf matrix before softmax.
  - GELU composed from primitives:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
  - tile_m = T (full sequence per tile) so all reshaping is dimensional,
    not data-movement, and avoids retiling between attention and MLP.
"""
import math

import torch
import torch.nn as nn
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

    eps_ld = _load_const(tile_m, M_tiles, eps, par_dispatch)
    s = BinaryMap(graph=graph, in1=var, in2=eps_ld,
                  fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    # rsqrt(s) via Pade + Newton-Raphson
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

    normalized = BinaryMap(graph=graph, in1=(xc_bc, 2), in2=y,
                           fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

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
    return BinaryMap(graph=graph, in1=scaled, in2=b_ld,
                     fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)


def _build_gelu(graph, x_stream, tile_m, M_tiles, par_dispatch,
                bw_mul, bw_add, bw_div):
    """GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)

    x_bc = Broadcast(graph=graph, input=x_stream, num_consumers=5)

    x_sq = BinaryMap(graph=graph, in1=(x_bc, 0), in2=(x_bc, 1),
                     fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    x_cube = BinaryMap(graph=graph, in1=x_sq, in2=(x_bc, 2),
                       fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

    c_coeff = _load_const(tile_m, M_tiles, 0.044715, par_dispatch)
    coeff_x3 = BinaryMap(graph=graph, in1=c_coeff, in2=x_cube,
                         fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    inner = BinaryMap(graph=graph, in1=(x_bc, 3), in2=coeff_x3,
                      fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    c_s2p = _load_const(tile_m, M_tiles, sqrt_2_over_pi, par_dispatch)
    z = BinaryMap(graph=graph, in1=c_s2p, in2=inner,
                  fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

    c_2 = _load_const(tile_m, M_tiles, 2.0, par_dispatch)
    two_z = BinaryMap(graph=graph, in1=c_2, in2=z,
                      fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
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

    c_1b = _load_const(tile_m, M_tiles, 1.0, par_dispatch)
    one_plus_tanh = BinaryMap(graph=graph, in1=c_1b, in2=tanh_z,
                              fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    c_05 = _load_const(tile_m, M_tiles, 0.5, par_dispatch)
    half_x = BinaryMap(graph=graph, in1=c_05, in2=(x_bc, 4),
                       fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    return BinaryMap(graph=graph, in1=half_x, in2=one_plus_tanh,
                     fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)


def _tree_add(graph, streams, bw_add):
    """Sum a list of streams via balanced binary tree of BinaryMap(Add)."""
    assert len(streams) >= 1
    while len(streams) > 1:
        next_level = []
        for i in range(0, len(streams), 2):
            if i + 1 < len(streams):
                s = BinaryMap(graph=graph, in1=streams[i], in2=streams[i + 1],
                              fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
                next_level.append(s)
            else:
                next_level.append(streams[i])
        streams = next_level
    return streams[0]


def _linear(graph, x, weight, bias_1d, tile_m, M_tiles, par_dispatch, bw_matmul, bw_add):
    """y = x @ W^T + b.  W shape (out, in), stored as nn.Linear convention."""
    out_dim = weight.shape[0]
    in_dim = weight.shape[1]
    w_load = OffChipLoad(underlying=weight, stride=(0, 0),
                         out_shape_tiled=(M_tiles, 1),
                         tile_row=out_dim, tile_col=in_dim, par_dispatch=par_dispatch)
    mm = BinaryMap(graph=graph, in1=x, in2=w_load,
                   fn=map_fn.Matmul(weight_transposed=True),
                   write_back_mu=False, compute_bw=bw_matmul)
    b_load = OffChipLoad(underlying=bias_1d.unsqueeze(0).contiguous(), stride=(0, 0),
                         out_shape_tiled=(M_tiles, 1),
                         tile_row=1, tile_col=out_dim, par_dispatch=par_dispatch)
    return BinaryMap(graph=graph, in1=mm, in2=b_load,
                     fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)


def build_graph(dims):
    B = dims["batch_size"]
    T = dims["seq_len"]
    C = dims["n_embd"]
    nh = dims["n_head"]
    max_seqlen = dims["max_seqlen"]
    hs = C // nh
    four_C = 4 * C

    tile_m = T       # full sequence per tile — avoids retiling between phases
    M_tiles = B      # one tile per batch element
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
    x_2d = x_3d.reshape(B * T, C).contiguous()

    # Extract attention weights
    W_attn = c_attn.weight.detach()       # (3C, C)
    b_attn = c_attn.bias.detach()         # (3C,)
    W_q, W_k, W_v = W_attn[:C], W_attn[C:2*C], W_attn[2*C:]
    b_q, b_k, b_v = b_attn[:C], b_attn[C:2*C], b_attn[2*C:]

    W_proj = c_proj_attn.weight.detach()  # (C, C)
    b_proj_attn_1d = c_proj_attn.bias.detach()
    W_proj_T = W_proj.T.contiguous()      # (C, C) = weight^T

    # Causal mask: 0 on/below diagonal, -inf above
    causal_mask = torch.zeros(T, T, dtype=torch.float32)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(T, T), diagonal=1).bool(), float('-inf'))

    # ===== Load x: stream (B, 1) of (T, C) =====
    x_load = OffChipLoad(
        underlying=x_2d, stride=(1, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=tile_m, tile_col=C, par_dispatch=par_dispatch,
    )

    # x -> (0: LN1, 1: residual_1)
    x_bc = Broadcast(graph=graph, input=x_load, num_consumers=2)

    # ===== LayerNorm 1 =====
    x_ln = _build_layernorm(
        graph, (x_bc, 0), ln_1.weight.detach(), ln_1.bias.detach(),
        tile_m, M_tiles, C, par_dispatch, bw_mul, bw_add, bw_div,
    )

    # ===== Per-head causal self-attention =====
    # Broadcast x_ln to nh heads
    x_ln_bc = Broadcast(graph=graph, input=x_ln, num_consumers=nh)

    head_projs = []
    for h in range(nh):
        # Broadcast to Q, K, V projections
        h_bc = Broadcast(graph=graph, input=(x_ln_bc, h), num_consumers=3)

        # --- Q, K, V projections: (T, C) @ (hs, C)^T -> (T, hs) + bias ---
        Wqh = W_q[h*hs:(h+1)*hs, :].contiguous()
        bqh = b_q[h*hs:(h+1)*hs]
        q_h = _linear(graph, (h_bc, 0), Wqh, bqh,
                       tile_m, M_tiles, par_dispatch, bw_matmul, bw_add)

        Wkh = W_k[h*hs:(h+1)*hs, :].contiguous()
        bkh = b_k[h*hs:(h+1)*hs]
        k_h = _linear(graph, (h_bc, 1), Wkh, bkh,
                       tile_m, M_tiles, par_dispatch, bw_matmul, bw_add)

        Wvh = W_v[h*hs:(h+1)*hs, :].contiguous()
        bvh = b_v[h*hs:(h+1)*hs]
        v_h = _linear(graph, (h_bc, 2), Wvh, bvh,
                       tile_m, M_tiles, par_dispatch, bw_matmul, bw_add)

        # --- QK^T / sqrt(hs): (T, hs) @ (T, hs)^T -> (T, T) ---
        qkt = BinaryMap(graph=graph, in1=q_h, in2=k_h,
                        fn=map_fn.Matmul(weight_transposed=True),
                        write_back_mu=False, compute_bw=bw_matmul)

        scale_ld = _load_const(tile_m, M_tiles, 1.0 / math.sqrt(hs), par_dispatch)
        scaled = BinaryMap(graph=graph, in1=qkt, in2=scale_ld,
                           fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

        # --- Causal mask (additive) ---
        mask_ld = OffChipLoad(underlying=causal_mask, stride=(0, 0),
                              out_shape_tiled=(M_tiles, 1),
                              tile_row=T, tile_col=T, par_dispatch=par_dispatch)
        masked = BinaryMap(graph=graph, in1=scaled, in2=mask_ld,
                           fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

        # --- Softmax (simplified, no max-subtract) ---
        exp_att = UnaryMap(graph=graph, input=masked,
                           fn=map_fn.Exp(), write_back_mu=False, compute_bw=bw_mul)
        exp_bc = Broadcast(graph=graph, input=exp_att, num_consumers=2)
        rowsum = UnaryMap(graph=graph, input=(exp_bc, 1),
                          fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=bw_div)
        softmax_out = BinaryMap(graph=graph, in1=(exp_bc, 0), in2=rowsum,
                                fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)

        # --- att @ V: (T, T) @ (T, hs) -> (T, hs) ---
        attn_out = BinaryMap(graph=graph, in1=softmax_out, in2=v_h,
                             fn=map_fn.Matmul(),
                             write_back_mu=False, compute_bw=bw_matmul)

        # --- Per-head output projection: (T, hs) @ (hs, C) -> (T, C) ---
        Wph_T = W_proj_T[h*hs:(h+1)*hs, :].contiguous()  # (hs, C)
        wp_ld = OffChipLoad(underlying=Wph_T, stride=(0, 0),
                            out_shape_tiled=(M_tiles, 1),
                            tile_row=hs, tile_col=C, par_dispatch=par_dispatch)
        proj_h = BinaryMap(graph=graph, in1=attn_out, in2=wp_ld,
                           fn=map_fn.Matmul(),
                           write_back_mu=False, compute_bw=bw_matmul)
        head_projs.append(proj_h)

    # ===== Sum head projections + bias =====
    attn_sum = _tree_add(graph, head_projs, bw_add)

    bp_ld = OffChipLoad(underlying=b_proj_attn_1d.unsqueeze(0).contiguous(),
                        stride=(0, 0), out_shape_tiled=(M_tiles, 1),
                        tile_row=1, tile_col=C, par_dispatch=par_dispatch)
    attn_biased = BinaryMap(graph=graph, in1=attn_sum, in2=bp_ld,
                            fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    # ===== Residual 1: x2 = attn_out + x =====
    x2 = BinaryMap(graph=graph, in1=attn_biased, in2=(x_bc, 1),
                   fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    # x2 -> (0: LN2, 1: residual_2)
    x2_bc = Broadcast(graph=graph, input=x2, num_consumers=2)

    # ===== LayerNorm 2 =====
    x_ln2 = _build_layernorm(
        graph, (x2_bc, 0), ln_2.weight.detach(), ln_2.bias.detach(),
        tile_m, M_tiles, C, par_dispatch, bw_mul, bw_add, bw_div,
    )

    # ===== MLP: FC1 -> GELU -> FC2 =====
    fc1 = _linear(graph, x_ln2,
                  c_fc.weight.detach().contiguous(), c_fc.bias.detach(),
                  tile_m, M_tiles, par_dispatch, bw_matmul, bw_add)

    gelu_out = _build_gelu(graph, fc1, tile_m, M_tiles, par_dispatch,
                           bw_mul, bw_add, bw_div)

    fc2 = _linear(graph, gelu_out,
                  c_proj_mlp.weight.detach().contiguous(), c_proj_mlp.bias.detach(),
                  tile_m, M_tiles, par_dispatch, bw_matmul, bw_add)

    # ===== Residual 2: output = fc2 + x2 =====
    output = BinaryMap(graph=graph, in1=fc2, in2=(x2_bc, 1),
                       fn=map_fn.Add(), write_back_mu=True, compute_bw=bw_add)

    output_op = OffChipStore(graph=graph, input=output,
                             par_dispatch=par_dispatch, store_file_name="output")

    graph = infer_broadcast(graph)
    return graph, output_op
