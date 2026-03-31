"""Multi-layer MLP baseline in STeP IR.

Chain of (Linear → ReLU) layers followed by a final Linear.
ReLU is not a STeP primitive, so we precompute binary masks on
the host (deterministic from SEED) and multiply element-wise.

Flow per hidden layer:
  1. BinaryMap(Matmul, weight_transposed=True) — y = x @ W^T
  2. BinaryMap(Add) — y += bias
  3. BinaryMap(Mul) with precomputed mask — ReLU(y) = y * (y > 0)

Final layer is Linear only (no ReLU).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx import MultiDiGraph

from step_py.functions import map_fn
from step_py.ops import OffChipLoad, BinaryMap, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def build_graph(dims):
    batch_size = dims["batch_size"]
    input_size = dims["input_size"]
    layer_sizes = dims["layer_sizes"]
    output_size = dims["output_size"]

    tile_m = batch_size
    M_tiles = 1
    par_dispatch = 4
    bw_matmul = 512
    bw_add = 128
    bw_mul = 128

    graph = MultiDiGraph()

    # ===== Replicate model weights (same RNG as compute_gold) =====
    torch.manual_seed(SEED)
    linears = []
    current_in = input_size
    for ls in layer_sizes:
        linears.append(nn.Linear(current_in, ls))
        current_in = ls
    linears.append(nn.Linear(current_in, output_size))

    # ===== Replicate input =====
    torch.manual_seed(SEED)
    x = torch.randn(batch_size, input_size, dtype=torch.float32)

    # ===== Precompute ReLU masks =====
    relu_masks = []
    with torch.no_grad():
        h = x
        for i in range(len(layer_sizes)):
            h = F.linear(h, linears[i].weight, linears[i].bias)
            mask = (h > 0).float()
            relu_masks.append(mask)
            h = h * mask

    # ===== Load input: stream (1, 1) of (batch_size, input_size) =====
    current_stream = OffChipLoad(
        underlying=x,
        stride=(1, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=tile_m, tile_col=input_size,
        par_dispatch=par_dispatch,
    )

    # ===== Hidden layers: Linear + ReLU =====
    for i, ls in enumerate(layer_sizes):
        W = linears[i].weight.detach().contiguous()   # (out, in)
        b = linears[i].bias.detach()                   # (out,)
        in_dim = W.shape[1]

        w_load = OffChipLoad(
            underlying=W, stride=(0, 0),
            out_shape_tiled=(M_tiles, 1),
            tile_row=ls, tile_col=in_dim,
            par_dispatch=par_dispatch,
        )
        mm = BinaryMap(
            graph=graph, in1=current_stream, in2=w_load,
            fn=map_fn.Matmul(weight_transposed=True),
            write_back_mu=False, compute_bw=bw_matmul,
        )

        b_load = OffChipLoad(
            underlying=b.unsqueeze(0).contiguous(), stride=(0, 0),
            out_shape_tiled=(M_tiles, 1),
            tile_row=1, tile_col=ls,
            par_dispatch=par_dispatch,
        )
        biased = BinaryMap(
            graph=graph, in1=mm, in2=b_load,
            fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add,
        )

        mask_load = OffChipLoad(
            underlying=relu_masks[i].contiguous(), stride=(1, 0),
            out_shape_tiled=(M_tiles, 1),
            tile_row=tile_m, tile_col=ls,
            par_dispatch=par_dispatch,
        )
        current_stream = BinaryMap(
            graph=graph, in1=biased, in2=mask_load,
            fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul,
        )

    # ===== Final linear (no ReLU) =====
    last = linears[-1]
    W = last.weight.detach().contiguous()
    b = last.bias.detach()

    w_load = OffChipLoad(
        underlying=W, stride=(0, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=output_size, tile_col=W.shape[1],
        par_dispatch=par_dispatch,
    )
    mm = BinaryMap(
        graph=graph, in1=current_stream, in2=w_load,
        fn=map_fn.Matmul(weight_transposed=True),
        write_back_mu=False, compute_bw=bw_matmul,
    )

    b_load = OffChipLoad(
        underlying=b.unsqueeze(0).contiguous(), stride=(0, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=1, tile_col=output_size,
        par_dispatch=par_dispatch,
    )
    output = BinaryMap(
        graph=graph, in1=mm, in2=b_load,
        fn=map_fn.Add(), write_back_mu=True, compute_bw=bw_add,
    )

    output_op = OffChipStore(
        graph=graph, input=output,
        par_dispatch=par_dispatch, store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
