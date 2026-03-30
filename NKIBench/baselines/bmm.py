"""bmm: C[b] = A[b] @ B[b] — batched matrix multiply."""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_accum_fn, init_fn
from step_py.ops import OffChipLoad, BinaryMapAccum, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def compute_gold(dims):
    B, M, K, N = dims["B"], dims["M"], dims["K"], dims["N"]
    np.random.seed(SEED)
    lhs = np.random.normal(0, 1.0, (B, M, K)).astype(np.float32)
    rhs = np.random.normal(0, 1.0, (B, K, N)).astype(np.float32)
    return torch.from_numpy(np.matmul(lhs, rhs)).float()


def build_graph(dims):
    B, M, K, N = dims["B"], dims["M"], dims["K"], dims["N"]
    tile_m, tile_k, tile_n = 64, 64, 64
    par_dispatch = 4
    compute_bw = 4096

    M_tiles = M // tile_m
    K_tiles = K // tile_k
    N_tiles = N // tile_n

    graph = MultiDiGraph()

    np.random.seed(SEED)
    A = torch.from_numpy(
        np.random.normal(0, 1.0, (B, M, K)).astype(np.float32).reshape(B * M, K)
    )
    B_mat = torch.from_numpy(
        np.random.normal(0, 1.0, (B, K, N)).astype(np.float32).reshape(B * K, N)
    )

    # A: (B*M, K) -> stream (B, M_tiles, N_tiles, K_tiles)
    a_load = OffChipLoad(
        underlying=A,
        stride=(M_tiles * K_tiles, K_tiles, 0, 1),
        out_shape_tiled=(B, M_tiles, N_tiles, K_tiles),
        tile_row=tile_m, tile_col=tile_k, par_dispatch=par_dispatch,
    )
    # B: (B*K, N) -> stream (B, M_tiles, N_tiles, K_tiles)
    b_load = OffChipLoad(
        underlying=B_mat,
        stride=(K_tiles * N_tiles, 0, 1, N_tiles),
        out_shape_tiled=(B, M_tiles, N_tiles, K_tiles),
        tile_row=tile_k, tile_col=tile_n, par_dispatch=par_dispatch,
    )

    matmul = BinaryMapAccum(
        graph=graph, in1=a_load, in2=b_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1, write_back_mu=True, compute_bw=compute_bw,
    )

    output_op = OffChipStore(
        graph=graph, input=matmul,
        par_dispatch=par_dispatch, store_file_name="output",
    )
    graph = infer_broadcast(graph)
    return graph, output_op
