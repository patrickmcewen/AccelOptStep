"""lora: y = x@w + x@a@b = x@(w + a@b)

LoRA adds a low-rank update to the base weight matrix. Since w, a, b are all
fixed model weights (not input-dependent), we fuse the LoRA weights:
    w_eff = w + a @ b
This is standard LoRA weight merging. The STeP graph then does a single GEMM.
"""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_accum_fn, init_fn
from step_py.ops import OffChipLoad, BinaryMapAccum, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def compute_gold(dims):
    M, K, N, R = dims["M"], dims["K"], dims["N"], dims["R"]
    np.random.seed(SEED)
    x = np.random.normal(0, 1.0, (M, K)).astype(np.float32)
    w = np.random.normal(0, 1.0, (K, N)).astype(np.float32)
    a = np.random.normal(0, 1.0, (K, R)).astype(np.float32)
    b = np.random.normal(0, 1.0, (R, N)).astype(np.float32)
    y1 = np.matmul(x, w)
    y2 = np.matmul(np.matmul(x, a), b)
    return torch.from_numpy(y1 + y2).float()


def build_graph(dims):
    M, K, N, R = dims["M"], dims["K"], dims["N"], dims["R"]
    tile_m, tile_k, tile_n = 64, 64, 64
    par_dispatch = 4
    compute_bw = 4096

    M_tiles = M // tile_m
    K_tiles = K // tile_k
    N_tiles = N // tile_n

    graph = MultiDiGraph()

    np.random.seed(SEED)
    x = np.random.normal(0, 1.0, (M, K)).astype(np.float32)
    w = np.random.normal(0, 1.0, (K, N)).astype(np.float32)
    a = np.random.normal(0, 1.0, (K, R)).astype(np.float32)
    b = np.random.normal(0, 1.0, (R, N)).astype(np.float32)

    # Merge LoRA weights: w_eff = w + a @ b
    w_eff = torch.from_numpy(w + np.matmul(a, b))
    X = torch.from_numpy(x)

    a_load = OffChipLoad(
        underlying=X, stride=(K_tiles, 0, 1),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=tile_m, tile_col=tile_k, par_dispatch=par_dispatch,
    )
    b_load = OffChipLoad(
        underlying=w_eff, stride=(0, 1, N_tiles),
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
