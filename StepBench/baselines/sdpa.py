import math

import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32, Tile
from step_py.functions import map_accum_fn, map_fn, accum_fn, init_fn
from step_py.ops import (
    OffChipLoad, BinaryMapAccum, BinaryMap, UnaryMap,
    Accum, Broadcast, OffChipStore, Reshape,
)
from rewrite.broadcast import infer_broadcast


SEED = 42


def build_graph(dims):
    """Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V.

    Q, K, V: (batch=32, heads=32, seq=512, dim=1024).
    Outer dimension = batch * heads = 1024.
    tile_seq=64, tile_dim=128.

    Simplified softmax: uses Exp (no max-subtract for numerical stability).
    For the Exp(QK^T) @ V step, V is loaded with tile_col=dim (full head dimension
    in one tile column) so that seq_k can be the accumulation (last) dimension.
    """
    batch = dims["batch"]
    heads = dims["heads"]
    seq = dims["seq"]
    dim = dims["dim"]
    outer = batch * heads
    tile_seq = 64
    tile_dim = 128
    par_dispatch = 4
    # Total compute bandwidth budget = 4096, distributed across 6 compute ops:
    #   qkt matmul = 1024, exp = 256, mult_v matmul = 1024,
    #   tile_wise_rowsum = 512, intra_tile_rowsum = 256, softmax_div = 1024
    # sum = 4096
    qkt_bw = 1024
    exp_bw = 256
    mult_v_bw = 1024
    rowsum_accum_bw = 512
    rowsum_map_bw = 256
    div_bw = 1024

    seq_tiles = seq // tile_seq
    dim_tiles = dim // tile_dim

    graph = MultiDiGraph()

    torch.manual_seed(SEED)
    Q_tensor = torch.randn(outer, seq, dim, dtype=torch.float32) / math.sqrt(dim)
    K_tensor = torch.randn(outer, seq, dim, dtype=torch.float32)
    # V stored as 2D (outer*seq, dim) so we can tile with tile_col=dim
    V_tensor_2d = torch.randn(outer * seq, dim, dtype=torch.float32)

    # ===== Step 1: QK^T = sum_l Q_{i,l} @ K_{j,l}^T =====
    # Q underlying tiled: (outer, seq_tiles, dim_tiles)
    # out_shape: (outer, seq_q_tiles, seq_k_tiles_broadcast, dim_tiles)
    q_load = OffChipLoad(
        underlying=Q_tensor,
        stride=(seq_tiles * dim_tiles, dim_tiles, 0, 1),
        out_shape_tiled=(outer, seq_tiles, seq_tiles, dim_tiles),
        tile_row=tile_seq,
        tile_col=tile_dim,
        par_dispatch=par_dispatch,
    )
    # stream shape: (1, outer, seq_tiles, seq_tiles, dim_tiles), tile: (tile_seq, tile_dim)

    # K underlying tiled: (outer, seq_tiles, dim_tiles)
    # out_shape: (outer, seq_q_tiles_broadcast, seq_k_tiles, dim_tiles)
    k_load = OffChipLoad(
        underlying=K_tensor,
        stride=(seq_tiles * dim_tiles, 0, dim_tiles, 1),
        out_shape_tiled=(outer, seq_tiles, seq_tiles, dim_tiles),
        tile_row=tile_seq,
        tile_col=tile_dim,
        par_dispatch=par_dispatch,
    )
    # stream shape: (1, outer, seq_tiles, seq_tiles, dim_tiles), tile: (tile_seq, tile_dim)

    # Accumulate over dim_tiles (rank=1)
    # Matmul(weight_transposed=True): (tile_seq, tile_dim) @ (tile_seq, tile_dim)^T -> (tile_seq, tile_seq)
    qkt = BinaryMapAccum(
        graph=graph,
        in1=q_load,
        in2=k_load,
        fn=map_accum_fn.Matmul(weight_transposed=True),
        init_fn=init_fn.Zero(shape=(tile_seq, tile_seq), dtype=Float32()),
        rank=1,
        write_back_mu=False,
        compute_bw=qkt_bw,
    )
    # stream: (outer, seq_q_tiles, seq_k_tiles), tile (tile_seq, tile_seq)

    # ===== Step 2: Exp(QK^T) =====
    exp_qkt = UnaryMap(
        graph=graph,
        input=qkt,
        fn=map_fn.Exp(),
        write_back_mu=False,
        compute_bw=exp_bw,
    )

    # Broadcast to 2 consumers: Exp@V and row sums
    exp_broadcast = Broadcast(graph=graph, input=exp_qkt, num_consumers=2)

    # ===== Step 3: Exp(QK^T) @ V =====
    # V loaded as 2D with tile_col=dim (full head dim per tile column).
    # V_tensor_2d tiled: (outer*seq_tiles, 1). dim//dim=1.
    # out_shape: (outer, seq_q_broadcast, seq_k_tiles)
    v_load = OffChipLoad(
        underlying=V_tensor_2d,
        stride=(seq_tiles, 0, 1),
        out_shape_tiled=(outer, seq_tiles, seq_tiles),
        tile_row=tile_seq,
        tile_col=dim,
        par_dispatch=par_dispatch,
    )
    # stream: (1, outer, seq_q_tiles, seq_k_tiles), tile (tile_seq, dim)

    # Accumulate over seq_k (rank=1)
    # Matmul: (tile_seq, tile_seq) @ (tile_seq, dim) -> (tile_seq, dim)
    mult_v = BinaryMapAccum(
        graph=graph,
        in1=(exp_broadcast, 0),
        in2=v_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_seq, dim), dtype=Float32()),
        rank=1,
        write_back_mu=False,
        compute_bw=mult_v_bw,
    )
    # stream: (outer, seq_q_tiles), tile (tile_seq, dim)

    # ===== Step 4: Row sums of Exp(QK^T) for softmax denominator =====
    # Accum over seq_k (rank=1)
    tile_wise_rowsum = Accum(
        graph=graph,
        input=(exp_broadcast, 1),
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_seq, tile_seq)),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(shape=(tile_seq, tile_seq), dtype=Float32()),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=rowsum_accum_bw,
    )
    # stream: (outer, seq_q_tiles), tile (tile_seq, tile_seq)

    # Intra-tile row-wise sum: (tile_seq, tile_seq) -> (tile_seq, 1)
    intra_tile_rowsum = UnaryMap(
        graph=graph,
        input=tile_wise_rowsum,
        fn=map_fn.RowWiseSum(),
        write_back_mu=False,
        compute_bw=rowsum_map_bw,
    )
    # stream: (outer, seq_q_tiles), tile (tile_seq, 1)

    # ===== Step 5: Divide by softmax denominator =====
    # (tile_seq, dim) / (tile_seq, 1) -> (tile_seq, dim)
    softmax_out = BinaryMap(
        graph=graph,
        in1=mult_v,
        in2=intra_tile_rowsum,
        fn=map_fn.Div(),
        write_back_mu=True,
        compute_bw=div_bw,
    )

    # ===== Step 6: Reshape stream to (outer, seq_q_tiles, 1) =====
    # Without this, the store lays out seq_q tiles as columns, producing
    # (outer*tile_seq, seq_tiles*dim) which has wrong element ordering.
    # Splitting the last dim with chunk_size=1 makes tensor_shape_tiled
    # (outer, seq_q_tiles, 1) so the store outputs (outer, seq*tile_seq, dim).
    reshaped = Reshape(
        graph=graph, input=softmax_out,
        chunk_size=1, reshape_rank=0, write_back_mu=False,
    )

    # ===== Step 7: Store output =====
    output_op = OffChipStore(
        graph=graph,
        input=reshaped,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
