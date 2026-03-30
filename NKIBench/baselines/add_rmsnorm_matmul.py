"""add_rmsnorm_matmul: rmsnorm(x + z, eps) * g  @  w

y = x + z
normalized = y / sqrt(mean(y^2) + eps) * g
output = normalized @ w

Single-graph using Bufferize/Streamify to chain rmsnorm output into matmul.
"""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_fn, map_accum_fn, init_fn
from step_py.ops import (
    OffChipLoad, UnaryMap, BinaryMap, BinaryMapAccum,
    Broadcast, Bufferize, Streamify, OffChipStore,
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
    z = np.random.normal(0, 1.0, (M, K)).astype(np.float32)
    g = np.random.normal(0, 1.0, (K,)).astype(np.float32)

    y = x + z
    t = np.square(y)
    t = np.divide(t, y.shape[-1])
    t = np.sum(t, axis=-1, keepdims=True)
    t = (t + eps).astype(y.dtype)
    y = y / np.sqrt(t)
    y = y * g
    return torch.from_numpy(np.matmul(y, w)).float()


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
    tile_n = 64
    par_dispatch = 4
    bw = 200

    M_tiles = M // tile_m
    N_tiles = N // tile_n

    graph = MultiDiGraph()

    np.random.seed(SEED)
    x_np = np.random.normal(0, 1.0, (M, K)).astype(np.float32)
    w_np = np.random.normal(0, 1.0, (K, N)).astype(np.float32)
    z_np = np.random.normal(0, 1.0, (M, K)).astype(np.float32)
    g_np = np.random.normal(0, 1.0, (K,)).astype(np.float32)

    x_t = torch.from_numpy(x_np)
    w_t = torch.from_numpy(w_np)
    z_t = torch.from_numpy(z_np)
    g_row = torch.from_numpy(g_np).unsqueeze(0).expand(tile_m, K).contiguous()

    # Full-row tiles: tile_col = K
    x_load = OffChipLoad(
        underlying=x_t, stride=(1, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=tile_m, tile_col=K, par_dispatch=par_dispatch,
    )
    z_load = OffChipLoad(
        underlying=z_t, stride=(1, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=tile_m, tile_col=K, par_dispatch=par_dispatch,
    )

    # y = x + z
    y = BinaryMap(graph=graph, in1=x_load, in2=z_load,
                  fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)

    # y broadcast to: (0) final normalization, (1,2) squaring
    y_bc = Broadcast(graph=graph, input=y, num_consumers=3)

    # y^2
    y_sq = BinaryMap(graph=graph, in1=(y_bc, 1), in2=(y_bc, 2),
                     fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)

    # mean(y^2) = RowWiseSum(y^2) / K
    sum_sq = UnaryMap(graph=graph, input=y_sq,
                      fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=bw)
    k_const = _load_const(tile_m, M_tiles, float(K), par_dispatch)
    mean_sq = BinaryMap(graph=graph, in1=sum_sq, in2=k_const,
                        fn=map_fn.Div(), write_back_mu=False, compute_bw=bw)

    # mean_sq + eps
    eps_load = _load_const(tile_m, M_tiles, eps, par_dispatch)
    s = BinaryMap(graph=graph, in1=mean_sq, in2=eps_load,
                  fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)

    # rsqrt via Padé + NR
    s_bc = Broadcast(graph=graph, input=s, num_consumers=NR_ITERS + 2)
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
    rsqrt_y = UnaryMap(graph=graph, input=neg_ratio,
                       fn=map_fn.Exp(), write_back_mu=False, compute_bw=bw)

    for i in range(NR_ITERS):
        c15 = _load_const(tile_m, M_tiles, 1.5, par_dispatch)
        cn05 = _load_const(tile_m, M_tiles, -0.5, par_dispatch)
        rsqrt_y = _nr_rsqrt_iter(graph, rsqrt_y, (s_bc, i + 2), c15, cn05, bw)

    # normalized = y * rsqrt
    normalized = BinaryMap(graph=graph, in1=(y_bc, 0), in2=rsqrt_y,
                           fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)

    # scaled = normalized * g
    g_load = OffChipLoad(
        underlying=g_row, stride=(0, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=tile_m, tile_col=K, par_dispatch=par_dispatch,
    )
    scaled = BinaryMap(graph=graph, in1=normalized, in2=g_load,
                       fn=map_fn.Mul(), write_back_mu=True, compute_bw=bw)
    # Stream: (M_tiles, 1), tile (tile_m, K)

    # --- Chain into matmul via Bufferize/Streamify ---
    buff = Bufferize(graph, scaled, rank=1)
    norm_expanded = Streamify(graph, buff, repeat_factor=[N_tiles], rank=1)
    # Stream: (M_tiles, N_tiles, 1), Tile(tile_m, K)

    K_tiles = 1
    w_load = OffChipLoad(
        underlying=w_t,
        stride=(0, 1, 0),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=K, tile_col=tile_n, par_dispatch=par_dispatch,
    )

    matmul = BinaryMapAccum(
        graph=graph, in1=norm_expanded, in2=w_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1, write_back_mu=True, compute_bw=2048,
    )

    output_op = OffChipStore(
        graph=graph, input=matmul,
        par_dispatch=par_dispatch, store_file_name="output",
    )
    graph = infer_broadcast(graph)
    return graph, output_op
