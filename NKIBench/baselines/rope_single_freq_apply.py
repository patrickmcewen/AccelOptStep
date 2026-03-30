"""rope_single_freq_apply: Rotary Position Embedding

x_out_0 = x0 * cos - x1 * sin
x_out_1 = x0 * sin + x1 * cos
x_out = concat([x_out_0, x_out_1], axis=0)

where x0 = x[:D/2, :], x1 = x[D/2:, :]
Input: x(D, B*H*N), cos(D/2, B*H*N), sin(D/2, B*H*N)

Strategy: arrange as (2, D/2, BHN) tiles. Load x0 and x1 with stride=0
over the "2" dimension, and arrange cos/sin/neg_sin into combined tensors
so that a uniform Mul+Add produces the rotated result.
"""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.functions import map_fn
from step_py.ops import OffChipLoad, BinaryMap, Broadcast, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def compute_gold(dims):
    B, H, N, D = dims["B"], dims["H"], dims["N"], dims["D"]
    BHN = B * H * N
    half_h = D // 2

    np.random.seed(SEED)
    x = np.random.normal(0, 1.0, (D, BHN)).astype(np.float32)
    freqs_cos = np.random.normal(0, 1.0, (half_h, BHN)).astype(np.float32)
    freqs_sin = np.random.normal(0, 1.0, (half_h, BHN)).astype(np.float32)

    x0 = x[:half_h, :]
    x1 = x[half_h:, :]
    x_out_0 = x0 * freqs_cos - x1 * freqs_sin
    x_out_1 = x0 * freqs_sin + x1 * freqs_cos
    x_out = np.concatenate([x_out_0, x_out_1], axis=0)
    return torch.from_numpy(x_out).float()


def build_graph(dims):
    B, H, N, D = dims["B"], dims["H"], dims["N"], dims["D"]
    BHN = B * H * N
    half_d = D // 2

    tile_r = half_d  # full half-D in one tile row
    tile_c = 64
    par_dispatch = 4

    assert BHN % tile_c == 0
    C_tiles = BHN // tile_c

    graph = MultiDiGraph()

    np.random.seed(SEED)
    x_np = np.random.normal(0, 1.0, (D, BHN)).astype(np.float32)
    cos_np = np.random.normal(0, 1.0, (half_d, BHN)).astype(np.float32)
    sin_np = np.random.normal(0, 1.0, (half_d, BHN)).astype(np.float32)

    # x0 = x[:half_d, :], x1 = x[half_d:, :]
    x0_t = torch.from_numpy(np.ascontiguousarray(x_np[:half_d, :]))
    x1_t = torch.from_numpy(np.ascontiguousarray(x_np[half_d:, :]))

    # Combined tensors for uniform computation across the "2" dim:
    # cos_sin[0,:] = cos, cos_sin[1,:] = sin  (both D/2 x BHN)
    # neg_sin_cos[0,:] = -sin, neg_sin_cos[1,:] = cos
    cos_sin = torch.from_numpy(
        np.concatenate([cos_np, sin_np], axis=0).copy()
    )  # (D, BHN)
    neg_sin_cos = torch.from_numpy(
        np.concatenate([-sin_np, cos_np], axis=0).copy()
    )  # (D, BHN)

    # Stream: (2, C_tiles), tile (half_d, tile_c)
    # x0 repeated for both stream positions (stride=0 for dim 0)
    x0_load = OffChipLoad(
        underlying=x0_t, stride=(0, 1),
        out_shape_tiled=(2, C_tiles),
        tile_row=tile_r, tile_col=tile_c, par_dispatch=par_dispatch,
    )
    # x1 repeated for both stream positions
    x1_load = OffChipLoad(
        underlying=x1_t, stride=(0, 1),
        out_shape_tiled=(2, C_tiles),
        tile_row=tile_r, tile_col=tile_c, par_dispatch=par_dispatch,
    )
    # cos_sin: position (0,n) = cos tile, position (1,n) = sin tile
    cs_load = OffChipLoad(
        underlying=cos_sin, stride=(C_tiles, 1),
        out_shape_tiled=(2, C_tiles),
        tile_row=tile_r, tile_col=tile_c, par_dispatch=par_dispatch,
    )
    # neg_sin_cos: position (0,n) = -sin tile, position (1,n) = cos tile
    nsc_load = OffChipLoad(
        underlying=neg_sin_cos, stride=(C_tiles, 1),
        out_shape_tiled=(2, C_tiles),
        tile_row=tile_r, tile_col=tile_c, par_dispatch=par_dispatch,
    )

    # term1 = x0 * cos_sin  (at pos 0: x0*cos, at pos 1: x0*sin)
    term1 = BinaryMap(
        graph=graph, in1=x0_load, in2=cs_load,
        fn=map_fn.Mul(), write_back_mu=False, compute_bw=1024,
    )
    # term2 = x1 * neg_sin_cos  (at pos 0: x1*(-sin), at pos 1: x1*cos)
    term2 = BinaryMap(
        graph=graph, in1=x1_load, in2=nsc_load,
        fn=map_fn.Mul(), write_back_mu=False, compute_bw=1024,
    )
    # out = term1 + term2  (at pos 0: x0*cos - x1*sin, at pos 1: x0*sin + x1*cos)
    result = BinaryMap(
        graph=graph, in1=term1, in2=term2,
        fn=map_fn.Add(), write_back_mu=True, compute_bw=2048,
    )

    output_op = OffChipStore(
        graph=graph, input=result,
        par_dispatch=par_dispatch, store_file_name="output",
    )
    graph = infer_broadcast(graph)
    return graph, output_op
