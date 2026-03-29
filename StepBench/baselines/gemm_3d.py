# StepBench/baselines/gemm_3d.py
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_accum_fn, init_fn
from step_py.ops import OffChipLoad, BinaryMapAccum, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def build_graph(dims):
    """3D tensor-matrix multiply: C = A @ B where A:(N,M,K), B:(K,L) -> C:(N,M,L)."""
    N_dim = dims["N"]
    M, K, L = dims["M"], dims["K"], dims["L"]

    tile_m, tile_k, tile_l = 64, 64, 64
    par_dispatch = 4
    compute_bw = 4096

    M_tiles = M // tile_m
    K_tiles = K // tile_k
    L_tiles = L // tile_l

    graph = MultiDiGraph()

    # Replicate RNG from compute_gold: manual_seed(42) -> randn(N,M,K) -> randn(K,L)
    torch.manual_seed(SEED)
    A = torch.randn(N_dim, M, K, dtype=torch.float32).reshape(N_dim * M, K)
    B = torch.randn(K, L, dtype=torch.float32)

    # A: (N*M, K), tiled as (N*M_tiles, K_tiles)
    # out_shape: (N, M_tiles, L_tiles, K_tiles) — broadcast over L
    a_load = OffChipLoad(
        underlying=A,
        stride=(M_tiles * K_tiles, K_tiles, 0, 1),
        out_shape_tiled=(N_dim, M_tiles, L_tiles, K_tiles),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )

    # B: (K, L) shared across all N — tiled as (K_tiles, L_tiles)
    # out_shape: (N, M_tiles, L_tiles, K_tiles) — broadcast over N and M
    b_load = OffChipLoad(
        underlying=B,
        stride=(0, 0, 1, L_tiles),
        out_shape_tiled=(N_dim, M_tiles, L_tiles, K_tiles),
        tile_row=tile_k,
        tile_col=tile_l,
        par_dispatch=par_dispatch,
    )

    matmul = BinaryMapAccum(
        graph=graph,
        in1=a_load,
        in2=b_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_l), dtype=Float32()),
        rank=1,
        write_back_mu=True,
        compute_bw=compute_bw,
    )

    output_op = OffChipStore(
        graph=graph,
        input=matmul,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
