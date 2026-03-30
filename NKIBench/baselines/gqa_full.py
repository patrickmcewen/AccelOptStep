"""gqa_full: Grouped Query Attention

q(B,N,QH,D), k(B,N,KH,D), v(B,N,KH,D)
n_rep = QH // KH
k_expanded = repeat(k, n_rep, axis=2)
After transpose: q(B,QH,N,D), k(B,QH,N,D), v(B,QH,N,D)

attention = softmax((q @ k^T) / sqrt(D), axis=-1)
output = attention @ v

This is multi-head SDPA with KV sharing across groups.
Adapted from StepBench/baselines/sdpa.py.
"""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32, Tile
from step_py.functions import map_accum_fn, map_fn, accum_fn, init_fn
from step_py.ops import (
    OffChipLoad, BinaryMapAccum, BinaryMap, UnaryMap,
    Accum, Broadcast, OffChipStore,
)
from rewrite.broadcast import infer_broadcast

SEED = 42


def compute_gold(dims):
    B_dim, N, QH, KH, D = dims["B"], dims["N"], dims["QH"], dims["KH"], dims["D"]
    np.random.seed(SEED)
    q = np.random.normal(0, 1.0, (B_dim, N, QH, D)).astype(np.float32)
    k = np.random.normal(0, 1.0, (B_dim, N, KH, D)).astype(np.float32)
    v = np.random.normal(0, 1.0, (B_dim, N, KH, D)).astype(np.float32)

    n_rep = QH // KH
    xk = np.repeat(k, n_rep, axis=2)
    xv = np.repeat(v, n_rep, axis=2)
    xq = q.transpose(0, 2, 1, 3)   # (B, QH, N, D)
    xk = xk.transpose(0, 2, 1, 3)  # (B, QH, N, D)
    xv = xv.transpose(0, 2, 1, 3)  # (B, QH, N, D)

    attention = (xq @ xk.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(D))
    exp_attention = np.exp(attention - np.max(attention, axis=-1, keepdims=True))
    attention = exp_attention / np.sum(exp_attention, axis=-1, keepdims=True)
    output = attention @ xv
    return torch.from_numpy(output).float()


def build_graph(dims):
    B_dim, N, QH, KH, D = dims["B"], dims["N"], dims["QH"], dims["KH"], dims["D"]
    n_rep = QH // KH
    outer = B_dim * QH
    tile_seq = min(64, N)
    tile_dim = min(64, D)
    par_dispatch = 4

    seq_tiles = N // tile_seq
    dim_tiles = D // tile_dim

    graph = MultiDiGraph()

    np.random.seed(SEED)
    q_np = np.random.normal(0, 1.0, (B_dim, N, QH, D)).astype(np.float32)
    k_np = np.random.normal(0, 1.0, (B_dim, N, KH, D)).astype(np.float32)
    v_np = np.random.normal(0, 1.0, (B_dim, N, KH, D)).astype(np.float32)

    # Transpose to (B, heads, N, D) and flatten outer dim
    # Q: (B, QH, N, D) -> (B*QH, N, D)
    Q = torch.from_numpy(
        q_np.transpose(0, 2, 1, 3).reshape(outer, N, D).copy()
    ) / np.sqrt(D).item()

    # K: (B, KH, N, D) -> (B*KH, N, D)
    K_flat = torch.from_numpy(
        k_np.transpose(0, 2, 1, 3).reshape(B_dim * KH, N, D).copy()
    )

    # V: (B*KH, N, D) -> (B*KH*N, D) for full-dim tiles
    V_flat = torch.from_numpy(
        v_np.transpose(0, 2, 1, 3).reshape(B_dim * KH * N, D).copy()
    )

    # ===== Step 1: QK^T =====
    # Q: stream (outer, seq_q, seq_k, dim_tiles), tile (tile_seq, tile_dim)
    q_load = OffChipLoad(
        underlying=Q,
        stride=(seq_tiles * dim_tiles, dim_tiles, 0, 1),
        out_shape_tiled=(outer, seq_tiles, seq_tiles, dim_tiles),
        tile_row=tile_seq, tile_col=tile_dim, par_dispatch=par_dispatch,
    )

    # K: stream (outer, seq_q, seq_k, dim_tiles)
    # For GQA: K has B*KH heads. Head h in [0,QH) maps to K head h//n_rep.
    # K stride for outer dim: each KH head has seq_tiles*dim_tiles tiles.
    # For QH head h, we want K head h//n_rep. With linear stride,
    # we need stride_outer = (seq_tiles*dim_tiles) // n_rep for consecutive
    # Q heads to map to the same K head.
    # But integer division in stride doesn't work directly.
    # For the small preset: B=1, QH=2, KH=1, n_rep=2.
    # K has 1 head -> all Q heads use the same K. stride_outer = 0.
    k_outer_stride = (seq_tiles * dim_tiles) // n_rep
    k_load = OffChipLoad(
        underlying=K_flat,
        stride=(k_outer_stride, 0, dim_tiles, 1),
        out_shape_tiled=(outer, seq_tiles, seq_tiles, dim_tiles),
        tile_row=tile_seq, tile_col=tile_dim, par_dispatch=par_dispatch,
    )

    qkt = BinaryMapAccum(
        graph=graph, in1=q_load, in2=k_load,
        fn=map_accum_fn.Matmul(weight_transposed=True),
        init_fn=init_fn.Zero(shape=(tile_seq, tile_seq), dtype=Float32()),
        rank=1, write_back_mu=False, compute_bw=1024,
    )

    # ===== Step 2: Exp(QK^T) =====
    exp_qkt = UnaryMap(
        graph=graph, input=qkt,
        fn=map_fn.Exp(), write_back_mu=False, compute_bw=256,
    )
    exp_bc = Broadcast(graph=graph, input=exp_qkt, num_consumers=2)

    # ===== Step 3: Exp(QK^T) @ V =====
    # V: (B*KH*N, D) tiled as (B*KH*seq_tiles, dim_tiles=1) if tile_col=D
    # or (B*KH*seq_tiles, dim_tiles) if tile_col=tile_dim
    # Use tile_col = D for full-dim tiles
    v_outer_stride = (seq_tiles) // n_rep
    v_load = OffChipLoad(
        underlying=V_flat,
        stride=(v_outer_stride, 0, 1),
        out_shape_tiled=(outer, seq_tiles, seq_tiles),
        tile_row=tile_seq, tile_col=D, par_dispatch=par_dispatch,
    )

    mult_v = BinaryMapAccum(
        graph=graph, in1=(exp_bc, 0), in2=v_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_seq, D), dtype=Float32()),
        rank=1, write_back_mu=False, compute_bw=1024,
    )

    # ===== Step 4: Row sums for softmax denominator =====
    tile_wise_rowsum = Accum(
        graph=graph, input=(exp_bc, 1),
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_seq, tile_seq)),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(shape=(tile_seq, tile_seq), dtype=Float32()),
        accum_rank=1, write_back_mu=False, compute_bw=512,
    )
    intra_tile_rowsum = UnaryMap(
        graph=graph, input=tile_wise_rowsum,
        fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=256,
    )

    # ===== Step 5: Divide by softmax denominator =====
    softmax_out = BinaryMap(
        graph=graph, in1=mult_v, in2=intra_tile_rowsum,
        fn=map_fn.Div(), write_back_mu=True, compute_bw=1024,
    )

    output_op = OffChipStore(
        graph=graph, input=softmax_out,
        par_dispatch=par_dispatch, store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
