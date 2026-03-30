"""bmm_softmax: softmax(A @ B, axis=-1) where A(B,M,K), B(B,K,N) -> (B,M,N)

Uses full-N tiles so softmax row-wise reduction stays within a single tile.
Simplified softmax (no max-subtract) — mathematically identical.
"""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_accum_fn, map_fn, init_fn
from step_py.ops import (
    OffChipLoad, BinaryMapAccum, UnaryMap, BinaryMap,
    Broadcast, OffChipStore,
)
from rewrite.broadcast import infer_broadcast

SEED = 42


def compute_gold(dims):
    B, K, M, N = dims["B"], dims["K"], dims["M"], dims["N"]
    np.random.seed(SEED)
    lhs = np.random.normal(0, 1.0, (B, M, K)).astype(np.float32)
    rhs = np.random.normal(0, 1.0, (B, K, N)).astype(np.float32)
    x = np.matmul(lhs, rhs)
    max_x = np.max(x, axis=2, keepdims=True)
    exp_x = np.exp(x - max_x)
    sum_exp = np.sum(exp_x, axis=2, keepdims=True)
    return torch.from_numpy(exp_x / sum_exp).float()


def build_graph(dims):
    B, K, M, N = dims["B"], dims["K"], dims["M"], dims["N"]
    tile_m = 64
    tile_k = K  # full K in one tile for accumulation
    par_dispatch = 4

    M_tiles = M // tile_m
    K_tiles = 1  # full K in one tile

    graph = MultiDiGraph()

    np.random.seed(SEED)
    lhs_np = np.random.normal(0, 1.0, (B, M, K)).astype(np.float32)
    rhs_np = np.random.normal(0, 1.0, (B, K, N)).astype(np.float32)

    # Flatten batch: A(B*M, K), B_mat(B*K, N)
    A = torch.from_numpy(lhs_np.reshape(B * M, K))
    B_mat = torch.from_numpy(rhs_np.reshape(B * K, N))

    # Full-N tiles: tile_n = N, N_tiles = 1
    # Stream: (B, M_tiles, 1, K_tiles=1)
    a_load = OffChipLoad(
        underlying=A,
        stride=(M_tiles * K_tiles, K_tiles, 0, 1),
        out_shape_tiled=(B, M_tiles, 1, K_tiles),
        tile_row=tile_m, tile_col=tile_k, par_dispatch=par_dispatch,
    )
    b_load = OffChipLoad(
        underlying=B_mat,
        stride=(K_tiles * 1, 0, 1, 1),
        out_shape_tiled=(B, M_tiles, 1, K_tiles),
        tile_row=tile_k, tile_col=N, par_dispatch=par_dispatch,
    )

    # Matmul: (tile_m, K) @ (K, N) -> (tile_m, N)
    matmul = BinaryMapAccum(
        graph=graph, in1=a_load, in2=b_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, N), dtype=Float32()),
        rank=1, write_back_mu=False, compute_bw=2048,
    )
    # Stream: (B, M_tiles, 1), tile (tile_m, N)

    # Exp
    exp_x = UnaryMap(
        graph=graph, input=matmul,
        fn=map_fn.Exp(), write_back_mu=False, compute_bw=512,
    )

    exp_bc = Broadcast(graph=graph, input=exp_x, num_consumers=2)

    # RowWiseSum: (tile_m, N) -> (tile_m, 1)
    rowsum = UnaryMap(
        graph=graph, input=(exp_bc, 1),
        fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=512,
    )

    # Div: (tile_m, N) / (tile_m, 1) -> (tile_m, N) via broadcast
    softmax_out = BinaryMap(
        graph=graph, in1=(exp_bc, 0), in2=rowsum,
        fn=map_fn.Div(), write_back_mu=True, compute_bw=1024,
    )

    output_op = OffChipStore(
        graph=graph, input=softmax_out,
        par_dispatch=par_dispatch, store_file_name="output",
    )
    graph = infer_broadcast(graph)
    return graph, output_op
