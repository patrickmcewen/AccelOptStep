import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32, Tile
from step_py.functions import map_accum_fn, map_fn, init_fn
from step_py.ops import OffChipLoad, BinaryMapAccum, UnaryMap, OffChipStore
from rewrite.broadcast import infer_broadcast


#M, K, N = 2048, 4096, 4096
M, K, N = 256, 256, 256
SEED = 42


def compute_gold():
    """PyTorch reference: y = SiLU(x @ W)."""
    torch.manual_seed(SEED)
    x = torch.randn(M, K, dtype=torch.float32)
    W = torch.randn(K, N, dtype=torch.float32)
    return torch.nn.functional.silu(x @ W)


def build_graph():
    """Fused GEMM+SiLU: y = SiLU(x @ W), x is (2048, 4096), W is (4096, 4096)."""
    tile_m, tile_k, tile_n = 128, 128, 128
    par_dispatch = 4
    compute_bw = 4096

    graph = MultiDiGraph()

    torch.manual_seed(SEED)
    x = torch.randn(M, K, dtype=torch.float32)
    W = torch.randn(K, N, dtype=torch.float32)

    # Load x: tiled as (M//tile_m, N//tile_n, K//tile_k) tiles of (tile_m, tile_k)
    x_load = OffChipLoad(
        underlying=x,
        stride=(K // tile_k, 0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )

    # Load W: tiled as (M//tile_m, N//tile_n, K//tile_k) tiles of (tile_k, tile_n)
    w_load = OffChipLoad(
        underlying=W,
        stride=(0, 1, N // tile_n),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=par_dispatch,
    )

    # Matmul accumulation over K — keep result on-chip (write_back_mu=False)
    matmul = BinaryMapAccum(
        graph=graph,
        in1=x_load,
        in2=w_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,
        write_back_mu=False,
        compute_bw=compute_bw,
    )

    # SiLU activation fused on-chip, write back to memory unit
    silu = UnaryMap(
        graph=graph,
        input=matmul,
        fn=map_fn.Silu(),
        write_back_mu=True,
        compute_bw=compute_bw,
    )

    # Store output
    output_op = OffChipStore(
        graph=graph,
        input=silu,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
