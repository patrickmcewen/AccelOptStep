# StepBench/baselines/gemm_swish_scaling.py
import torch
import torch.nn as nn
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_accum_fn, map_fn, init_fn
from step_py.ops import OffChipLoad, BinaryMapAccum, UnaryMap, BinaryMap, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def build_graph(dims):
    """Fused GEMM+Swish+Scaling: y = SiLU(x @ W^T) * s.

    Matches problems/gemm_swish_scaling.py compute_gold() which uses
    nn.Linear(bias=False) with Kaiming-uniform weights.
    """
    M = dims["batch_size"]
    K = dims["in_features"]
    N = dims["out_features"]
    scaling_factor = dims["scaling_factor"]

    tile_m, tile_k, tile_n = 128, 128, 128
    par_dispatch = 4
    # Total compute bandwidth budget = 4096, distributed across 3 compute ops:
    #   matmul (critical path) = 2048, silu = 1024, scaling mul = 1024
    matmul_bw = 2048
    silu_bw = 1024
    scale_bw = 1024

    graph = MultiDiGraph()

    # Replicate the exact RNG sequence from compute_gold():
    #   torch.manual_seed(42) -> nn.Linear(K, N, bias=False) -> manual_seed(42) -> randn(M, K)
    torch.manual_seed(SEED)
    linear = nn.Linear(K, N, bias=False)
    W = linear.weight.detach()  # (N, K) — nn.Linear stores weights transposed

    torch.manual_seed(SEED)
    x = torch.randn(M, K, dtype=torch.float32)
    scale = torch.full((M, N), scaling_factor, dtype=torch.float32)

    # x: (M, K), tiled as (M//tm, N//tn, K//tk) with broadcast over N
    x_load = OffChipLoad(
        underlying=x,
        stride=(K // tile_k, 0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )

    # W: (N, K) used with weight_transposed=True
    # Matmul computes (tile_m, tile_k) @ (tile_n, tile_k)^T -> (tile_m, tile_n)
    w_load = OffChipLoad(
        underlying=W,
        stride=(0, K // tile_k, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_n,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )

    matmul = BinaryMapAccum(
        graph=graph,
        in1=x_load,
        in2=w_load,
        fn=map_accum_fn.Matmul(weight_transposed=True),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,
        write_back_mu=False,
        compute_bw=matmul_bw,
    )

    silu = UnaryMap(
        graph=graph,
        input=matmul,
        fn=map_fn.Silu(),
        write_back_mu=False,
        compute_bw=silu_bw,
    )

    scale_load = OffChipLoad(
        underlying=scale,
        stride=(N // tile_n, 1),
        out_shape_tiled=(M // tile_m, N // tile_n),
        tile_row=tile_m,
        tile_col=tile_n,
        par_dispatch=par_dispatch,
    )

    scaled = BinaryMap(
        graph=graph,
        in1=silu,
        in2=scale_load,
        fn=map_fn.Mul(),
        write_back_mu=True,
        compute_bw=scale_bw,
    )

    output_op = OffChipStore(
        graph=graph,
        input=scaled,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
