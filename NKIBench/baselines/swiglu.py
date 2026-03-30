"""swiglu: SwiGLU activation with matmul

up = x @ w_up          (M, N)
gate = silu(x @ w_gate) (M, N)
output = (gate * up) @ w_down  (M, K)

Single-graph using Bufferize/Streamify to chain the intermediate into the
final matmul without writing to HBM.
"""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_fn, map_accum_fn, init_fn
from step_py.ops import (
    OffChipLoad, UnaryMap, BinaryMap, BinaryMapAccum,
    Broadcast, Bufferize, Streamify, OffChipStore,
)
from rewrite.broadcast import infer_broadcast

SEED = 42


def compute_gold(dims):
    M, K, N = dims["M"], dims["K"], dims["N"]
    np.random.seed(SEED)
    x = np.random.normal(0, 1.0, (M, K)).astype(np.float32)
    w_up = np.random.normal(0, 1.0, (K, N)).astype(np.float32)
    w_down = np.random.normal(0, 1.0, (N, K)).astype(np.float32)
    w_gate = np.random.normal(0, 1.0, (K, N)).astype(np.float32)

    up = np.matmul(x, w_up)
    gate = np.matmul(x, w_gate)
    activated = gate / (1 + np.exp(-gate))  # silu
    return torch.from_numpy(np.matmul(activated * up, w_down)).float()


def build_graph(dims):
    M, K, N = dims["M"], dims["K"], dims["N"]
    tile_m, tile_k, tile_n = 64, 64, 64
    par_dispatch = 4

    M_tiles = M // tile_m
    K_in_tiles = K // tile_k
    N_tiles = N // tile_n
    K_out_tiles = K // tile_k  # output dim of final matmul

    graph = MultiDiGraph()

    np.random.seed(SEED)
    x_np = np.random.normal(0, 1.0, (M, K)).astype(np.float32)
    w_up_np = np.random.normal(0, 1.0, (K, N)).astype(np.float32)
    w_down_np = np.random.normal(0, 1.0, (N, K)).astype(np.float32)
    w_gate_np = np.random.normal(0, 1.0, (K, N)).astype(np.float32)

    x_t = torch.from_numpy(x_np)
    w_up_t = torch.from_numpy(w_up_np)
    w_down_t = torch.from_numpy(w_down_np)
    w_gate_t = torch.from_numpy(w_gate_np)

    # x broadcast to 2 matmul paths
    x_load = OffChipLoad(
        underlying=x_t, stride=(K_in_tiles, 0, 1),
        out_shape_tiled=(M_tiles, N_tiles, K_in_tiles),
        tile_row=tile_m, tile_col=tile_k, par_dispatch=par_dispatch,
    )
    x_bc = Broadcast(graph=graph, input=x_load, num_consumers=2)

    # Path 1: x @ w_up -> up
    w_up_load = OffChipLoad(
        underlying=w_up_t, stride=(0, 1, N_tiles),
        out_shape_tiled=(M_tiles, N_tiles, K_in_tiles),
        tile_row=tile_k, tile_col=tile_n, par_dispatch=par_dispatch,
    )
    up = BinaryMapAccum(
        graph=graph, in1=(x_bc, 0), in2=w_up_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1, write_back_mu=False, compute_bw=1500,
    )
    # Stream: (M_tiles, N_tiles), tile (tile_m, tile_n)

    # Path 2: silu(x @ w_gate) -> activated_gate
    w_gate_load = OffChipLoad(
        underlying=w_gate_t, stride=(0, 1, N_tiles),
        out_shape_tiled=(M_tiles, N_tiles, K_in_tiles),
        tile_row=tile_k, tile_col=tile_n, par_dispatch=par_dispatch,
    )
    gate_mm = BinaryMapAccum(
        graph=graph, in1=(x_bc, 1), in2=w_gate_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1, write_back_mu=False, compute_bw=1500,
    )
    gate_silu = UnaryMap(
        graph=graph, input=gate_mm,
        fn=map_fn.Silu(), write_back_mu=False, compute_bw=256,
    )

    # intermediate = gate_silu * up: (M_tiles, N_tiles), tile (tile_m, tile_n)
    inter = BinaryMap(
        graph=graph, in1=gate_silu, in2=up,
        fn=map_fn.Mul(), write_back_mu=True, compute_bw=256,
    )

    # --- Chain into final matmul via Bufferize/Streamify ---
    # Bufferize: absorb N_tiles (the reduction dimension for the next matmul)
    buff = Bufferize(graph, inter, rank=1)
    # Stream: (M_tiles,), Buffer(Tile(tile_m, tile_n), shape=(N_tiles,))

    # Streamify: replay for each K_out tile position
    inter_expanded = Streamify(graph, buff, repeat_factor=[K_out_tiles], rank=1)
    # Stream: (M_tiles, K_out_tiles, N_tiles), Tile(tile_m, tile_n)

    # Load w_down with matching shape: (M_tiles, K_out_tiles, N_tiles)
    # w_down is (N, K): tile_row=tile_n, tile_col=tile_k
    # Tiled as (N_tiles, K_out_tiles)
    w_down_load = OffChipLoad(
        underlying=w_down_t,
        stride=(0, 1, K_out_tiles),
        out_shape_tiled=(M_tiles, K_out_tiles, N_tiles),
        tile_row=tile_n, tile_col=tile_k, par_dispatch=par_dispatch,
    )

    # Matmul: (tile_m, tile_n) @ (tile_n, tile_k) -> (tile_m, tile_k)
    # Accumulate over N_tiles
    matmul = BinaryMapAccum(
        graph=graph, in1=inter_expanded, in2=w_down_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_k), dtype=Float32()),
        rank=1, write_back_mu=True, compute_bw=512,
    )
    # Stream: (M_tiles, K_out_tiles), tile (tile_m, tile_k)

    output_op = OffChipStore(
        graph=graph, input=matmul,
        par_dispatch=par_dispatch, store_file_name="output",
    )
    graph = infer_broadcast(graph)
    return graph, output_op
