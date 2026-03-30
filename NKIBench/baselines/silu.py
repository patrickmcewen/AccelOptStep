"""silu: y = x / (1 + exp(-x)) = SiLU(x)"""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.functions import map_fn
from step_py.ops import OffChipLoad, UnaryMap, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def compute_gold(dims):
    M, N = dims["M"], dims["N"]
    np.random.seed(SEED)
    x = np.random.normal(0, 1.0, (M, N)).astype(np.float32)
    return torch.from_numpy(x / (1 + np.exp(-x))).float()


def build_graph(dims):
    M, N = dims["M"], dims["N"]
    tile_m, tile_n = 64, 64
    par_dispatch = 4

    M_tiles = M // tile_m
    N_tiles = N // tile_n

    graph = MultiDiGraph()

    np.random.seed(SEED)
    x = torch.from_numpy(np.random.normal(0, 1.0, (M, N)).astype(np.float32))

    x_load = OffChipLoad(
        underlying=x, stride=(N_tiles, 1),
        out_shape_tiled=(M_tiles, N_tiles),
        tile_row=tile_m, tile_col=tile_n, par_dispatch=par_dispatch,
    )

    silu = UnaryMap(
        graph=graph, input=x_load,
        fn=map_fn.Silu(), write_back_mu=True, compute_bw=4096,
    )

    output_op = OffChipStore(
        graph=graph, input=silu,
        par_dispatch=par_dispatch, store_file_name="output",
    )
    graph = infer_broadcast(graph)
    return graph, output_op
