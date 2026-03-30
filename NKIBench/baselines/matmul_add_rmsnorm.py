"""matmul_add_rmsnorm: rmsnorm(x @ w + z) * g

y = x @ w + z
output = y * g / sqrt(mean(y^2) + eps)

Single-graph: use tile_n = N (full output width) for the matmul so that
rmsnorm reduction (RowWiseSum) works on full-row tiles in the same graph.
"""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_fn, map_accum_fn, init_fn
from step_py.ops import (
    OffChipLoad, UnaryMap, BinaryMap, BinaryMapAccum,
    Broadcast, OffChipStore,
)
from rewrite.broadcast import infer_broadcast

SEED = 42
NR_ITERS = 4


def compute_gold(dims):
    M, K, N = dims["M"], dims["K"], dims["N"]
    eps = 1e-5
    np.random.seed(SEED)
    x = np.random.normal(0, 1.0, (M, K)).astype(np.float32)
    w = np.random.normal(0, 1.0, (K, N)).astype(np.float32)
    # eps is scalar, skip
    z = np.random.normal(0, 1.0, (M, N)).astype(np.float32)
    g = np.random.normal(0, 1.0, (N,)).astype(np.float32)

    y = np.matmul(x, w) + z
    rms = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True) + eps)
    return torch.from_numpy(y * g / rms).float()


def _load_const(tile_m, M_tiles, val, par_dispatch):
    t = torch.full((tile_m, 1), val, dtype=torch.float32)
    return OffChipLoad(underlying=t, stride=(0, 0),
                       out_shape_tiled=(M_tiles, 1),
                       tile_row=tile_m, tile_col=1, par_dispatch=par_dispatch)


def _nr_rsqrt_iter(graph, y_prev, s, const_1_5, const_neg_0_5, bw):
    y_bc = Broadcast(graph=graph, input=y_prev, num_consumers=3)
    y_sq = BinaryMap(graph=graph, in1=(y_bc, 0), in2=(y_bc, 1),
                     fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)
    s_y_sq = BinaryMap(graph=graph, in1=s, in2=y_sq,
                       fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)
    neg_half = BinaryMap(graph=graph, in1=const_neg_0_5, in2=s_y_sq,
                         fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)
    bracket = BinaryMap(graph=graph, in1=const_1_5, in2=neg_half,
                        fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)
    y_new = BinaryMap(graph=graph, in1=(y_bc, 2), in2=bracket,
                      fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)
    return y_new


def build_graph(dims):
    M, K, N = dims["M"], dims["K"], dims["N"]
    eps = 1e-5
    tile_m = 64
    tile_k = 64
    par_dispatch = 4
    bw = 200

    M_tiles = M // tile_m
    K_tiles = K // tile_k

    graph = MultiDiGraph()

    np.random.seed(SEED)
    x_np = np.random.normal(0, 1.0, (M, K)).astype(np.float32)
    w_np = np.random.normal(0, 1.0, (K, N)).astype(np.float32)
    z_np = np.random.normal(0, 1.0, (M, N)).astype(np.float32)
    g_np = np.random.normal(0, 1.0, (N,)).astype(np.float32)

    x_t = torch.from_numpy(x_np)
    w_t = torch.from_numpy(w_np)
    z_t = torch.from_numpy(z_np)
    g_row = torch.from_numpy(g_np).unsqueeze(0).expand(tile_m, N).contiguous()

    # Matmul x @ w with tile_n = N (full output width)
    # Stream: (M_tiles, 1, K_tiles), tile (tile_m, tile_k) and (tile_k, N)
    x_load = OffChipLoad(
        underlying=x_t,
        stride=(K_tiles, 0, 1),
        out_shape_tiled=(M_tiles, 1, K_tiles),
        tile_row=tile_m, tile_col=tile_k, par_dispatch=par_dispatch,
    )
    w_load = OffChipLoad(
        underlying=w_t,
        stride=(0, 0, 1),
        out_shape_tiled=(M_tiles, 1, K_tiles),
        tile_row=tile_k, tile_col=N, par_dispatch=par_dispatch,
    )

    matmul = BinaryMapAccum(
        graph=graph, in1=x_load, in2=w_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, N), dtype=Float32()),
        rank=1, write_back_mu=False, compute_bw=2048,
    )
    # Stream: (M_tiles, 1), tile (tile_m, N)

    # y = matmul + z
    z_load = OffChipLoad(
        underlying=z_t, stride=(1, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=tile_m, tile_col=N, par_dispatch=par_dispatch,
    )
    y = BinaryMap(graph=graph, in1=matmul, in2=z_load,
                  fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)

    # y broadcast to: (0) final normalization, (1,2) squaring
    y_bc = Broadcast(graph=graph, input=y, num_consumers=3)

    # y^2
    y_sq = BinaryMap(graph=graph, in1=(y_bc, 1), in2=(y_bc, 2),
                     fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)

    # mean(y^2) = RowWiseSum(y^2) / N
    sum_sq = UnaryMap(graph=graph, input=y_sq,
                      fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=bw)
    n_const = _load_const(tile_m, M_tiles, float(N), par_dispatch)
    mean_sq = BinaryMap(graph=graph, in1=sum_sq, in2=n_const,
                        fn=map_fn.Div(), write_back_mu=False, compute_bw=bw)

    # s = mean_sq + eps
    eps_load = _load_const(tile_m, M_tiles, eps, par_dispatch)
    s = BinaryMap(graph=graph, in1=mean_sq, in2=eps_load,
                  fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)

    # Range reduction for rsqrt: y = x@w + z has variance ≈ K, so mean(y²) ≈ K.
    # Scale s by 1/K to bring it near 1 for Padé+NR convergence, then adjust.
    # rsqrt(s) = rsqrt(s/K) * rsqrt(K) = rsqrt(s/K) / sqrt(K)
    scale_factor = float(K)
    k_scale = _load_const(tile_m, M_tiles, scale_factor, par_dispatch)
    s_scaled = BinaryMap(graph=graph, in1=s, in2=k_scale,
                         fn=map_fn.Div(), write_back_mu=False, compute_bw=bw)

    # rsqrt(s_scaled) via Padé + NR (s_scaled ≈ 1)
    s_bc = Broadcast(graph=graph, input=s_scaled, num_consumers=NR_ITERS + 2)
    neg1 = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    s_m1 = BinaryMap(graph=graph, in1=(s_bc, 0), in2=neg1,
                     fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)
    pos1 = _load_const(tile_m, M_tiles, 1.0, par_dispatch)
    s_p1 = BinaryMap(graph=graph, in1=(s_bc, 1), in2=pos1,
                     fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)
    ratio = BinaryMap(graph=graph, in1=s_m1, in2=s_p1,
                      fn=map_fn.Div(), write_back_mu=False, compute_bw=bw)
    neg1_2 = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    neg_ratio = BinaryMap(graph=graph, in1=neg1_2, in2=ratio,
                          fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)
    rsqrt_scaled = UnaryMap(graph=graph, input=neg_ratio,
                            fn=map_fn.Exp(), write_back_mu=False, compute_bw=bw)

    for i in range(NR_ITERS):
        c15 = _load_const(tile_m, M_tiles, 1.5, par_dispatch)
        cn05 = _load_const(tile_m, M_tiles, -0.5, par_dispatch)
        rsqrt_scaled = _nr_rsqrt_iter(graph, rsqrt_scaled, (s_bc, i + 2), c15, cn05, bw)

    # Recover rsqrt(s) = rsqrt(s_scaled) * rsqrt(K)
    rsqrt_k = _load_const(tile_m, M_tiles, 1.0 / np.sqrt(scale_factor), par_dispatch)
    rsqrt_s = BinaryMap(graph=graph, in1=rsqrt_scaled, in2=rsqrt_k,
                        fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)

    # normalized = y * rsqrt
    normalized = BinaryMap(graph=graph, in1=(y_bc, 0), in2=rsqrt_s,
                           fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)

    # output = normalized * g
    g_load = OffChipLoad(
        underlying=g_row, stride=(0, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=tile_m, tile_col=N, par_dispatch=par_dispatch,
    )
    result = BinaryMap(graph=graph, in1=normalized, in2=g_load,
                       fn=map_fn.Mul(), write_back_mu=True, compute_bw=bw)

    output_op = OffChipStore(
        graph=graph, input=result,
        par_dispatch=par_dispatch, store_file_name="output",
    )
    graph = infer_broadcast(graph)
    return graph, output_op
