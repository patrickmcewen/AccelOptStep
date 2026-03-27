# StepBench/baselines/gemm.py
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_accum_fn, init_fn
from step_py.ops import OffChipLoad, BinaryMapAccum, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def build_graph(dims):
    """Weight-stationary GEMM: C = A @ B."""
    M, K, N = dims["M"], dims["K"], dims["N"]
    tile_m, tile_k, tile_n = 128, 128, 128
    par_dispatch = 4
    compute_bw = 4096

    graph = MultiDiGraph()

    torch.manual_seed(SEED)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    a_load = OffChipLoad(
        underlying=A,
        stride=(K // tile_k, 0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )

    b_load = OffChipLoad(
        underlying=B,
        stride=(0, 1, N // tile_n),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
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
