# StepBench/baselines/rmsnorm.py
"""RMSNorm: x / sqrt(mean(x^2, dim=1, keepdim=True) + eps)

Input shape: (B, F, D1, D2). Reduction is along dim=1 (features, size F).

To avoid permutation (which breaks output memory order), we reshape to
(B*D1*D2, F) via view tricks that work because D1*D2 is contiguous within
each (b, f) slice, and we use a BinaryMapAccum matmul with ones(F,1) to
reduce across features.

Key insight: x_4d.permute(0,2,3,1).contiguous().reshape(-1, F) produces
(B*D1*D2, F). The STeP output in this layout reshapes back via
.reshape(B, D1, D2, F).permute(0,3,1,2) = (B, F, D1, D2).

Instead we work in the ORIGINAL memory layout (B*F*D1, D2) and use
tile_row = F*D1 to capture full feature-spatial blocks, then use a
matmul reduction with a weight matrix to sum across features.

Actually: reshape x to (B*D1*D2, F) and DON'T use contiguous — just
build the tensor data in the permuted order for OffChipLoad. The sim
output shape will be (B*D1*D2, 1) for the rsqrt, and the final multiply
produces (B*D1*D2, F). We then reshape this to (B, D1, D2, F) and the
test comparison handles the permutation.

Simplest correct approach: work in (B*D1*D2, F) layout, output in same
layout, and have the test reshape correctly. We add a custom reshape in
verify_baseline.
"""
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_fn, map_accum_fn, init_fn
from step_py.ops import OffChipLoad, UnaryMap, BinaryMap, Broadcast, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42
NR_ITERS = 4


def _load_const(tile_m, M_tiles, val, par_dispatch):
    t = torch.full((tile_m, 1), val, dtype=torch.float32)
    return OffChipLoad(underlying=t, stride=(0, 0),
                       out_shape_tiled=(M_tiles, 1),
                       tile_row=tile_m, tile_col=1, par_dispatch=par_dispatch)


def _nr_rsqrt_iter(graph, y_prev, s, const_1_5, const_neg_0_5, bw_mul, bw_add):
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


# Output permutation info for test comparison
OUTPUT_PERMUTE = (0, 3, 1, 2)  # (B, D1, D2, F) -> (B, F, D1, D2)
OUTPUT_INTERMEDIATE_SHAPE = None  # set dynamically


def build_graph(dims):
    B = dims["batch_size"]
    F = dims["features"]
    D1 = dims["dim1"]
    D2 = dims["dim2"]
    eps = 1e-5

    graph = MultiDiGraph()

    torch.manual_seed(SEED)
    x_4d = torch.randn(B, F, D1, D2, dtype=torch.float32)

    # Permute to (B, D1, D2, F) and flatten to (B*D1*D2, F)
    x_2d = x_4d.permute(0, 2, 3, 1).contiguous().reshape(-1, F)
    M = x_2d.shape[0]  # B*D1*D2

    # Ones vector for matmul-based feature reduction: (F, 1)
    ones_F = torch.ones(F, 1, dtype=torch.float32)

    tile_m = min(64, M)
    assert M % tile_m == 0
    M_tiles = M // tile_m
    par_dispatch = 4

    bw_mul = 150
    bw_add = 100
    bw_matmul = 200
    bw_div = 200
    bw_final = 200

    # Load x: (M_tiles, 1) tile (tile_m, F)
    x_load = OffChipLoad(underlying=x_2d, stride=(1, 0),
                         out_shape_tiled=(M_tiles, 1),
                         tile_row=tile_m, tile_col=F, par_dispatch=par_dispatch)

    # x → (0) numerator, (1,2) for x^2
    x_bc = Broadcast(graph=graph, input=x_load, num_consumers=3)

    # x^2: tile (tile_m, F)
    x_sq = BinaryMap(graph=graph, in1=(x_bc, 1), in2=(x_bc, 2),
                     fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

    # sum(x^2) across features via matmul: (tile_m, F) @ (F, 1) -> (tile_m, 1)
    # Using BinaryMapAccum with rank=1 to accumulate if F > tile_k
    # But F=8 fits in one tile, so rank=1 with 1 accumulation step
    ones_load = OffChipLoad(underlying=ones_F, stride=(0, 0),
                            out_shape_tiled=(M_tiles, 1, 1),
                            tile_row=F, tile_col=1, par_dispatch=par_dispatch)

    # Reshape x_sq stream to add K dim for matmul accumulation
    # x_sq has stream (M_tiles, 1) tile (tile_m, F)
    # ones has stream (M_tiles, 1, 1) tile (F, 1)
    # For BinaryMapAccum: need both to have same stream shape with K dim last
    # x_sq needs stream (M_tiles, 1, 1) tile (tile_m, F) — add K_tiles=1
    # Actually, OffChipLoad for x_sq already has the right shape, just need
    # the accumulation dimension. Let me restructure the loads.

    # Reload x_sq data won't work. Instead, use RowWiseSum for the reduction
    # since F fits in one tile column. RowWiseSum sums across columns of a tile.
    sum_x_sq = UnaryMap(graph=graph, input=x_sq,
                        fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=bw_matmul)

    # mean = sum / F: tile (tile_m, 1) / scalar F
    f_load = _load_const(tile_m, M_tiles, float(F), par_dispatch)
    mean_x_sq = BinaryMap(graph=graph, in1=sum_x_sq, in2=f_load,
                          fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)

    # mean + eps
    eps_load = _load_const(tile_m, M_tiles, eps, par_dispatch)
    s = BinaryMap(graph=graph, in1=mean_x_sq, in2=eps_load,
                  fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    # rsqrt(s) via Padé initial estimate + Newton-Raphson refinement.
    # Initial estimate: y0 = exp(-(s-1)/(s+1)) ≈ rsqrt(s) for s in [0.01, 10].
    # Then NR iterations refine to full precision.
    s_bc = Broadcast(graph=graph, input=s, num_consumers=NR_ITERS + 2)

    # y0 = exp(-(s-1)/(s+1))  — Padé approximant for rsqrt
    neg_ones_load = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    s_minus_1 = BinaryMap(graph=graph, in1=(s_bc, 0), in2=neg_ones_load,
                          fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    ones_load = _load_const(tile_m, M_tiles, 1.0, par_dispatch)
    s_plus_1 = BinaryMap(graph=graph, in1=(s_bc, 1), in2=ones_load,
                         fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    ratio = BinaryMap(graph=graph, in1=s_minus_1, in2=s_plus_1,
                      fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)
    neg_one_load2 = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    neg_ratio = BinaryMap(graph=graph, in1=neg_one_load2, in2=ratio,
                          fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    y = UnaryMap(graph=graph, input=neg_ratio,
                 fn=map_fn.Exp(), write_back_mu=False, compute_bw=bw_mul)

    for i in range(NR_ITERS):
        c15 = _load_const(tile_m, M_tiles, 1.5, par_dispatch)
        cn05 = _load_const(tile_m, M_tiles, -0.5, par_dispatch)
        y = _nr_rsqrt_iter(graph, y, (s_bc, i + 2), c15, cn05, bw_mul, bw_add)

    # output = x * rsqrt(s): tile (tile_m, F) * tile (tile_m, 1) broadcasts
    result = BinaryMap(graph=graph, in1=(x_bc, 0), in2=y,
                       fn=map_fn.Mul(), write_back_mu=True, compute_bw=bw_final)

    output_op = OffChipStore(graph=graph, input=result,
                             par_dispatch=par_dispatch, store_file_name="output")

    graph = infer_broadcast(graph)
    return graph, output_op
