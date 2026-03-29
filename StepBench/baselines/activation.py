# StepBench/baselines/activation.py
import torch
from networkx import MultiDiGraph

from step_py.datatype import Float32, Tile
from step_py.functions import map_fn, accum_fn, init_fn
from step_py.ops import OffChipLoad, UnaryMap, BinaryMap, Accum, Broadcast, RepeatStatic, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42

# Supported activation functions and their STeP implementations
SUPPORTED = {"swish", "softmax"}


def _build_swish(graph, x_tensor, M, N, tile_m, tile_n, par_dispatch):
    """x * sigmoid(x) = SiLU(x)."""
    M_tiles = M // tile_m
    N_tiles = N // tile_n

    x_load = OffChipLoad(
        underlying=x_tensor,
        stride=(N_tiles, 1),
        out_shape_tiled=(M_tiles, N_tiles),
        tile_row=tile_m,
        tile_col=tile_n,
        par_dispatch=par_dispatch,
    )

    silu = UnaryMap(
        graph=graph,
        input=x_load,
        fn=map_fn.Silu(),
        write_back_mu=True,
        compute_bw=4096,
    )
    return silu


def _build_softmax(graph, x_tensor, M, N, tile_m, tile_n, par_dispatch):
    """softmax(x, dim=1) = exp(x) / sum(exp(x), dim=1).

    Each tile spans one full row (tile_col=N) so RowWiseSum produces the
    complete per-row denominator without cross-tile accumulation. A dummy
    outer dimension of 1 is added to produce a 2D stream shape which the
    simulator requires.
    Simplified softmax without max-subtract (same as sdpa baseline).
    """
    M_tiles = M // tile_m

    # Bandwidth budget: exp=1024, rowsum=1024, div=2048
    exp_bw = 1024
    rowsum_bw = 1024
    div_bw = 2048

    # Use full-row tiles: tile_col = N, with a dummy trailing dim for 2D stream
    x_load = OffChipLoad(
        underlying=x_tensor,
        stride=(1, 0),
        out_shape_tiled=(M_tiles, 1),
        tile_row=tile_m,
        tile_col=N,
        par_dispatch=par_dispatch,
    )

    exp_x = UnaryMap(
        graph=graph,
        input=x_load,
        fn=map_fn.Exp(),
        write_back_mu=False,
        compute_bw=exp_bw,
    )

    # Broadcast exp to: (0) numerator path, (1) denominator path
    exp_broadcast = Broadcast(graph=graph, input=exp_x, num_consumers=2)

    # Denominator: row-wise sum within each tile
    rowsum = UnaryMap(
        graph=graph,
        input=(exp_broadcast, 1),
        fn=map_fn.RowWiseSum(),
        write_back_mu=False,
        compute_bw=rowsum_bw,
    )
    # stream: (1, M_tiles) tile (tile_m, 1) — same stream shape as numerator

    # Divide: exp(x) / row_sums — tile (tile_m, N) / tile (tile_m, 1) broadcasts
    softmax_out = BinaryMap(
        graph=graph,
        in1=(exp_broadcast, 0),
        in2=rowsum,
        fn=map_fn.Div(),
        write_back_mu=True,
        compute_bw=div_bw,
    )
    return softmax_out


def build_graph(dims):
    """Element-wise activation function."""
    M = dims["batch_size"]
    N = dims["dim"]
    fn_name = dims["fn"]

    assert fn_name in SUPPORTED, (
        f"Activation '{fn_name}' not supported in STeP baseline. "
        f"Supported: {SUPPORTED}"
    )

    tile_m, tile_n = 64, 64
    par_dispatch = 4

    graph = MultiDiGraph()

    torch.manual_seed(SEED)
    x = torch.randn(M, N, dtype=torch.float32)

    if fn_name == "swish":
        last_op = _build_swish(graph, x, M, N, tile_m, tile_n, par_dispatch)
    elif fn_name == "softmax":
        last_op = _build_softmax(graph, x, M, N, tile_m, tile_n, par_dispatch)

    output_op = OffChipStore(
        graph=graph,
        input=last_op,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
