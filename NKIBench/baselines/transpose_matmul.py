"""transpose_matmul: C = A^T @ B where A(K,M), B(K,N) -> C(M,N)"""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_accum_fn, init_fn
from step_py.ops import OffChipLoad, BinaryMapAccum, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def compute_gold(dims):
    M, K, N = dims["M"], dims["K"], dims["N"]
    np.random.seed(SEED)
    lhs = np.random.normal(loc=0, scale=1.0, size=(K, M)).astype(np.float32)
    rhs = np.random.normal(loc=0, scale=1.0, size=(K, N)).astype(np.float32)
    return torch.from_numpy(np.matmul(lhs.T, rhs)).float()


def build_graph(dims):
    M, K, N = dims["M"], dims["K"], dims["N"]
    tile_m, tile_k, tile_n = 64, 64, 64
    par_dispatch = 4
    compute_bw = 4096

    M_tiles = M // tile_m
    K_tiles = K // tile_k
    N_tiles = N // tile_n

    graph = MultiDiGraph()

    np.random.seed(SEED)
    lhs_raw = np.random.normal(0, 1.0, (K, M)).astype(np.float32)
    rhs_raw = np.random.normal(0, 1.0, (K, N)).astype(np.float32)

    # Transpose A to (M, K) for standard GEMM loading
    A = torch.from_numpy(np.ascontiguousarray(lhs_raw.T))
    B = torch.from_numpy(rhs_raw)

    a_load = OffChipLoad(
        underlying=A, stride=(K_tiles, 0, 1),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=tile_m, tile_col=tile_k, par_dispatch=par_dispatch,
    )
    b_load = OffChipLoad(
        underlying=B, stride=(0, 1, N_tiles),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
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
