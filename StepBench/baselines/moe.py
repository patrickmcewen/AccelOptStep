"""Simplified Mixture of Experts (MoE) baseline following the STeP paper (Section 3.3).

Top-1 routing with FlatPartition/FlatReassemble for dynamic token routing.
Each expert is a single GEMM. Tokens are packed into tiles for efficient
matrix-matrix multiplication, then unpacked and reassembled in original order.

Flow:
  1. Load input as row-per-tile stream, flatten
  2. Pre-compute routing (gate), create selector via SelectGen
  3. FlatPartition to route tokens to experts
  4. Pack rows into tiles (Reshape + Flatten + Accum RetileRow)
  5. EagerMerge expert streams for time-multiplexed weight loading
  6. ExpertAddrGen + RandomOffChipLoad for per-expert weights
  7. BinaryMap(Matmul) for computation
  8. FlatPartition to re-distribute results to per-expert streams
  9. RetileStreamify to unpack tiles back to rows
  10. FlatReassemble to merge in original token order
  11. Accum(Add) to collapse ragged dimension
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx import MultiDiGraph

from step_py.datatype import Float32, Uint64, Tile
from step_py.functions import map_fn, accum_fn, init_fn
from step_py.ops import (
    OffChipLoad, BinaryMap, Accum, Broadcast, EagerMerge, FlatPartition,
    FlatReassemble, Flatten, RandomOffChipLoad, RepeatStatic, Reshape,
    RetileStreamify, OffChipStore,
)
from step_py.utility_ops import SelectGen, ExpertAddrGen
from rewrite.broadcast import infer_broadcast

SEED = 42


def build_graph(dims):
    B = dims["num_tokens"]
    D = dims["hidden_dim"]
    N = dims["intermediate_dim"]
    E = dims["num_experts"]
    tile_N = dims.get("tile_N", 4)  # rows packed per tile

    par_dispatch = 4
    N_tiles = N // N  # = 1 (no N-tiling: entire N fits in one tile)
    assert N_tiles == 1, "Simplified MoE: intermediate_dim must be a single tile"

    # Bandwidth budget = 4096
    matmul_bw = 3072
    accum_bw = 512
    retile_bw = 512

    graph = MultiDiGraph()

    # ========== Replicate RNG sequence from compute_gold ==========
    torch.manual_seed(SEED)
    model = nn.Linear(D, E, bias=False)            # gate weights
    expert_linears = [nn.Linear(D, N, bias=False) for _ in range(E)]

    torch.manual_seed(SEED)
    x = torch.randn(B, D, dtype=torch.float32)

    # Pre-compute routing on host (deterministic)
    with torch.no_grad():
        gate_scores = F.linear(x, model.weight)    # (B, E)
        expert_idx = gate_scores.argmax(dim=-1)     # (B,)

    # Create multihot selector: (B, E) binary tensor (int64 for SelectGen)
    multihot = F.one_hot(expert_idx, num_classes=E)  # (B, E) int64

    # Expert weights transposed for BinaryMap(Matmul): (D, N) per expert
    # Stack all experts along dim 0: (E*D, N) for RandomOffChipLoad
    expert_weights_t = [lin.weight.T.detach().contiguous() for lin in expert_linears]
    stacked_weights = torch.cat(expert_weights_t, dim=0)  # (E*D, N)

    # ========== Stage 1: Load input ==========
    # (B, 1) of (1, D) tiles
    in_load = OffChipLoad(
        underlying=x,
        stride=(1, 1),
        out_shape_tiled=(B, 1),
        tile_row=1, tile_col=D,
        par_dispatch=par_dispatch,
    )
    # Flatten last dim: (B,) of (1, D)
    flat_in = Flatten(graph=graph, input=in_load, min_rank=0, max_rank=1)

    # ========== Stage 2: Routing selector ==========
    selector = SelectGen(is_multihot=True, tensor=multihot, n=E)
    # Stream: (B,) dtype=MultiHot(E)

    # ========== Stage 3: FlatPartition — route tokens to experts ==========
    partition = FlatPartition(
        graph=graph,
        input=flat_in,
        control=selector,
        partition_rank=0,
        switch_cycles=[1] * E,
        write_back_mu=False,
        num_consumers=E,
    )
    # Per expert i: (Di,) of (1, D) — dynamic number of tokens

    # ========== Stage 4: Pack rows into tiles ==========
    # Reshape: (Di,) → (1, ceil(Di/tile_N), tile_N) of (1, D)
    reshaped = [
        Reshape(
            graph=graph,
            input=(partition, i),
            chunk_size=tile_N,
            reshape_rank=0,
            write_back_mu=False,
            add_outer_dim=True,
            pad_fn=init_fn.Zero(shape=(1, D), dtype=Float32()),
        )
        for i in range(E)
    ]

    # Flatten outer dims: (ceil(Di/tile_N), tile_N) of (1, D)
    flat_reshaped = [
        Flatten(graph=graph, input=reshaped[i], min_rank=1, max_rank=2)
        for i in range(E)
    ]

    # Accum RetileRow: collect tile_N rows → (ceil(Di/tile_N),) of (tile_N, D)
    packed = [
        Accum(
            graph=graph,
            input=flat_reshaped[i],
            output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_N, D)),
            fn=accum_fn.RetileRow(),
            init_fn=init_fn.Empty(shape=(0, D), dtype=Float32()),
            accum_rank=1,
            write_back_mu=False,
            compute_bw=retile_bw,
        )
        for i in range(E)
    ]

    # ========== Stage 5: EagerMerge expert streams ==========
    merged = EagerMerge(graph=graph, inputs=packed, input_rank=0)
    # data: (sum(ceil(Di/tile_N)),) of (tile_N, D)
    # select: (sum(...),) MultiHot(E)

    # ========== Stage 6: Load expert weights ==========
    # RepeatStatic for N_tiles dimension (= 1 here)
    repeated = RepeatStatic(
        graph=graph,
        input=merged.data_tuple(),
        repeat_factor=N_tiles,
    )
    # → (sum(...), 1) of (tile_N, D)

    # ExpertAddrGen: generates addresses based on select stream
    addr_gen = ExpertAddrGen(
        graph=graph,
        input=merged.select_tuple(),
        num_tile_per_expert=N_tiles,
        expert_addr_base=0,
    )
    # → (sum(...), 1, 1) of address tiles

    # RandomOffChipLoad: load correct expert's weight
    # stacked_weights: (E*D, N), tiled as tile_row=D, tile_col=N
    # E tiles along dim 0 (one per expert), 1 tile along dim 1
    weight_load = RandomOffChipLoad(
        graph=graph,
        raddr=addr_gen,
        underlying=stacked_weights,
        tile_row=D,
        tile_col=N,
        base_addr_byte=0,
        par_dispatch=par_dispatch,
    )
    # → (sum(...), 1, 1) of (D, N)

    # Flatten last dim: (sum(...), 1) of (D, N)
    flat_weights = Flatten(
        graph=graph, input=weight_load, min_rank=0, max_rank=1,
    )

    # ========== Stage 7: Compute matmul ==========
    # (tile_N, D) × (D, N) → (tile_N, N)
    matmul_out = BinaryMap(
        graph=graph,
        in1=repeated,
        in2=flat_weights,
        fn=map_fn.Matmul(weight_transposed=False),
        write_back_mu=False,
        compute_bw=matmul_bw,
    )
    # → (sum(...), 1) of (tile_N, N)

    # Flatten N_tiles dim: (sum(...),) of (tile_N, N)
    flat_matmul = Flatten(
        graph=graph, input=matmul_out, min_rank=0, max_rank=1,
    )

    # ========== Stage 8: Re-distribute to per-expert ==========
    redistrib = FlatPartition(
        graph=graph,
        input=flat_matmul,
        control=merged.select_tuple(),
        partition_rank=0,
        switch_cycles=[1] * E,
        write_back_mu=False,
        num_consumers=E,
    )
    # Per expert: (ceil(Di/tile_N),) of (tile_N, N)

    # ========== Stage 9: Unpack tiles to rows ==========
    # RetileStreamify: (tile_N, N) → (1, N) tiles, expanding stream dimension
    unpacked = [
        RetileStreamify(
            graph=graph,
            input=(redistrib, i),
            split_row=True,
            filter_mask=True,
        )
        for i in range(E)
    ]
    # Per expert: (Dyn,) of (1, N) — filter_mask removes padding rows

    # Set dynamic dimensions to match original partition sizes
    for part_stream, retiled in zip(partition.stream_list, unpacked):
        retiled.stream.shape = (part_stream.shape[0],)

    # ========== Stage 10: FlatReassemble ==========
    # New selector for reassembly (same data, fresh stream)
    selector_reassemble = SelectGen(is_multihot=True, tensor=multihot, n=E)

    reassembled = FlatReassemble(
        graph=graph,
        inputs=unpacked,
        control=selector_reassemble,
        reassemble_rank=0,
        switch_cycles=[1] * E,
        write_back_mu=False,
    )
    # → (B, Ragged) of (1, N) — for top-1 routing, Ragged=1

    # ========== Stage 11: Accumulate ragged dimension ==========
    accumulated = Accum(
        graph=graph,
        input=reassembled,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(1, N)),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(shape=(1, N), dtype=Float32()),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=accum_bw,
    )
    # → (B,) of (1, N)

    # ========== Stage 12: Store ==========
    # Reshape to add dim for OffChipStore
    reshaped_out = Reshape(
        graph=graph,
        input=accumulated,
        chunk_size=1,
        reshape_rank=0,
        write_back_mu=True,
    )
    # → (B, 1) of (1, N)

    output_op = OffChipStore(
        graph=graph, input=reshaped_out,
        par_dispatch=par_dispatch, store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
