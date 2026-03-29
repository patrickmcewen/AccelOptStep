# StepBench/baselines/gemm_batched.py
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_accum_fn, init_fn
from step_py.ops import OffChipLoad, BinaryMapAccum, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def build_graph(dims):
    """Batched GEMM: C[b] = A[b] @ B[b] for each batch b."""
    batch = dims["batch"]
    M, K, N = dims["M"], dims["K"], dims["N"]

    tile_m, tile_k, tile_n = 64, 64, 64
    par_dispatch = 4
    compute_bw = 4096

    M_tiles = M // tile_m
    K_tiles = K // tile_k
    N_tiles = N // tile_n

    graph = MultiDiGraph()

    # Replicate RNG from compute_gold: manual_seed(42) -> randn(batch,M,K) -> randn(batch,K,N)
    torch.manual_seed(SEED)
    A = torch.randn(batch, M, K, dtype=torch.float32).reshape(batch * M, K)
    B = torch.randn(batch, K, N, dtype=torch.float32).reshape(batch * K, N)

    # A: (batch*M, K), tiled as (batch*M_tiles, K_tiles)
    # out_shape: (batch, M_tiles, N_tiles, K_tiles) — broadcast over N
    a_load = OffChipLoad(
        underlying=A,
        stride=(M_tiles * K_tiles, K_tiles, 0, 1),
        out_shape_tiled=(batch, M_tiles, N_tiles, K_tiles),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )

    # B: (batch*K, N), tiled as (batch*K_tiles, N_tiles)
    # out_shape: (batch, M_tiles, N_tiles, K_tiles) — broadcast over M
    b_load = OffChipLoad(
        underlying=B,
        stride=(K_tiles * N_tiles, 0, 1, N_tiles),
        out_shape_tiled=(batch, M_tiles, N_tiles, K_tiles),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=par_dispatch,
    )

    matmul = BinaryMapAccum(
        graph=graph,
        in1=a_load,
        in2=b_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
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
