# StepBench/baselines/gemm_scale_residual.py
"""Fused GEMM + Scaling + Residual: out = linear(x) * scaling_factor + linear(x)
                                       = linear(x) * (1 + scaling_factor)

Since linear(x) = x @ W^T + b, and Add is not supported as a BinaryMap in the
STeP simulator, we use augmented matrices to fold the bias into the matmul:
  x_aug = [x | 1]          shape (M, K+1)  padded to (M, K_pad)
  W_T_aug = [W^T; b]       shape (K+1, N)  padded to (K_pad, N)
  x_aug @ W_T_aug = x @ W^T + b

Then multiply the result by (1 + scaling_factor).
"""
import torch
import torch.nn as nn
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_accum_fn, map_fn, init_fn
from step_py.ops import OffChipLoad, BinaryMapAccum, BinaryMap, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def _ceil_to(x, m):
    """Round x up to the next multiple of m."""
    return ((x + m - 1) // m) * m


def build_graph(dims):
    """Fused GEMM+Scale+Residual: y = (x @ W^T + b) * (1 + scaling_factor)."""
    M = dims["batch_size"]
    K = dims["in_features"]
    N = dims["out_features"]
    scaling_factor = dims["scaling_factor"]

    tile_m, tile_k, tile_n = 64, 64, 64
    par_dispatch = 4
    matmul_bw = 3072
    scale_bw = 1024

    graph = MultiDiGraph()

    # Replicate the exact RNG sequence from compute_gold():
    # Model(in_features, out_features, scaling_factor) creates nn.Linear(K, N) with bias
    torch.manual_seed(SEED)
    linear = nn.Linear(K, N)
    W = linear.weight.detach()   # (N, K)
    b = linear.bias.detach()     # (N,)

    torch.manual_seed(SEED)
    x = torch.randn(M, K, dtype=torch.float32)

    # Augmented matrices: fold bias into matmul
    K_aug = K + 1
    K_pad = _ceil_to(K_aug, tile_k)

    # x_aug: (M, K_pad) — original x, then column of 1s, then zero padding
    x_aug = torch.zeros(M, K_pad, dtype=torch.float32)
    x_aug[:, :K] = x
    x_aug[:, K] = 1.0

    # W_T_aug: (K_pad, N) — W^T, then bias row, then zero padding
    W_T_aug = torch.zeros(K_pad, N, dtype=torch.float32)
    W_T_aug[:K, :] = W.T
    W_T_aug[K, :] = b

    # Pre-scale by (1 + scaling_factor)
    combined_scale = 1.0 + scaling_factor
    scale_tensor = torch.full((M, N), combined_scale, dtype=torch.float32)

    M_tiles = M // tile_m
    K_tiles = K_pad // tile_k
    N_tiles = N // tile_n

    # x_aug load: (M, K_pad), broadcast over N
    x_load = OffChipLoad(
        underlying=x_aug,
        stride=(K_tiles, 0, 1),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )

    # W_T_aug load: (K_pad, N), broadcast over M
    w_load = OffChipLoad(
        underlying=W_T_aug,
        stride=(0, 1, N_tiles),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=par_dispatch,
    )

    matmul = BinaryMapAccum(
        graph=graph,
        in1=x_load,
        in2=w_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,
        write_back_mu=False,
        compute_bw=matmul_bw,
    )

    # Multiply by (1 + scaling_factor)
    scale_load = OffChipLoad(
        underlying=scale_tensor,
        stride=(N_tiles, 1),
        out_shape_tiled=(M_tiles, N_tiles),
        tile_row=tile_m,
        tile_col=tile_n,
        par_dispatch=par_dispatch,
    )

    scaled = BinaryMap(
        graph=graph,
        in1=matmul,
        in2=scale_load,
        fn=map_fn.Mul(),
        write_back_mu=True,
        compute_bw=scale_bw,
    )

    output_op = OffChipStore(
        graph=graph,
        input=scaled,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
