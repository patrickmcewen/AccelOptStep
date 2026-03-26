import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32, Tile
from step_py.functions import map_accum_fn, init_fn
from step_py.ops import OffChipLoad, BinaryMapAccum, OffChipStore
from rewrite.broadcast import infer_broadcast


#M, K, N = 128, 1024, 1024
M, K, N = 256, 256, 256
SEED = 42


def compute_gold():
    """PyTorch reference: C = A @ B."""
    torch.manual_seed(SEED)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    return A @ B


def build_graph():
    """Weight-stationary GEMM: C = A @ B, where A, B are (4096, 4096) float32."""
    tile_m, tile_k, tile_n = 128, 128, 128
    par_dispatch = 4
    compute_bw = 4096

    graph = MultiDiGraph()

    torch.manual_seed(SEED)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    # Load A: tiled as (M//tile_m, N//tile_n, K//tile_k) tiles of (tile_m, tile_k)
    # stride=(K//tile_k, 0, 1) means: M-dim advances by K//tile_k tiles,
    # N-dim is broadcast (0), K-dim advances by 1
    a_load = OffChipLoad(
        underlying=A,
        stride=(K // tile_k, 0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )
    # a_load stream shape: (1, M//tile_m, N//tile_n, K//tile_k), tile: (tile_m, tile_k)

    # Load B: tiled as (M//tile_m, N//tile_n, K//tile_k) tiles of (tile_k, tile_n)
    # stride=(0, 1, N//tile_n) means: M-dim is broadcast (0),
    # N-dim advances by 1, K-dim advances by N//tile_n
    b_load = OffChipLoad(
        underlying=B,
        stride=(0, 1, N // tile_n),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=par_dispatch,
    )
    # b_load stream shape: (1, M//tile_m, N//tile_n, K//tile_k), tile: (tile_k, tile_n)

    # Matmul accumulation over the K dimension (rank=1 reduces last dim)
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
    # matmul stream shape: (M//tile_m, N//tile_n), tile: (tile_m, tile_n)

    # Store output
    output_op = OffChipStore(
        graph=graph,
        input=matmul,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
