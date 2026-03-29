# StepBench/baselines/layernorm.py
"""LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias

Input shape: (B, F, D1, D2). Normalization is over the last 3 dims (F, D1, D2).

Since F*D1*D2 is contiguous in memory for each batch element, we can reshape
x to (B, F*D1*D2) without permutation.  Full-row tiles (tile_col = F*D1*D2)
let RowWiseSum compute the complete per-row reduction in one shot.

nn.LayerNorm has learnable weight and bias of shape (F, D1, D2), flattened
to (F*D1*D2,) for the element-wise multiply and add.
"""
import torch
import torch.nn as nn
from networkx import MultiDiGraph

from step_py.datatype import Float32
from step_py.functions import map_fn, init_fn
from step_py.ops import OffChipLoad, UnaryMap, BinaryMap, Broadcast, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42
NR_ITERS = 4


def _load_const(tile_m, M_tiles, val, par_dispatch):
    """Load a scalar constant broadcast across all stream positions."""
    t = torch.full((tile_m, 1), val, dtype=torch.float32)
    return OffChipLoad(underlying=t, stride=(0, 0),
                       out_shape_tiled=(M_tiles, 1),
                       tile_row=tile_m, tile_col=1, par_dispatch=par_dispatch)


def _nr_rsqrt_iter(graph, y_prev, s, const_1_5, const_neg_0_5, bw_mul, bw_add):
    """One Newton-Raphson iteration: y_new = y * (1.5 + (-0.5)*s*y^2)."""
    y_bc = Broadcast(graph=graph, input=y_prev, num_consumers=3)
    y_sq = BinaryMap(graph=graph, in1=(y_bc, 0), in2=(y_bc, 1),
                     fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    s_y_sq = BinaryMap(graph=graph, in1=s, in2=y_sq,
                       fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    neg_half_s_y_sq = BinaryMap(graph=graph, in1=const_neg_0_5, in2=s_y_sq,
                                fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    bracket = BinaryMap(graph=graph, in1=const_1_5, in2=neg_half_s_y_sq,
                        fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    y_new = BinaryMap(graph=graph, in1=(y_bc, 2), in2=bracket,
                      fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    return y_new


def build_graph(dims):
    B = dims["batch_size"]
    F = dims["features"]
    D1 = dims["dim1"]
    D2 = dims["dim2"]
    eps = 1e-5
    N = F * D1 * D2  # number of elements per batch row

    graph = MultiDiGraph()

    # Replicate exact RNG sequence from compute_gold():
    #   torch.manual_seed(42)
    #   model = Model((F, D1, D2))   -> creates nn.LayerNorm with weight & bias
    #   inputs = get_inputs(dims)    -> torch.manual_seed(42); randn(...)
    torch.manual_seed(SEED)
    ln = nn.LayerNorm((F, D1, D2))
    weight = ln.weight.detach().reshape(-1)  # (N,) — ones by default
    bias = ln.bias.detach().reshape(-1)      # (N,) — zeros by default

    torch.manual_seed(SEED)
    x_4d = torch.randn(B, F, D1, D2, dtype=torch.float32)
    x_2d = x_4d.reshape(B, N)  # contiguous reshape, no permutation needed

    tile_m = B
    tile_col = N
    M_tiles = 1
    par_dispatch = 4

    bw_mul = 150
    bw_add = 100
    bw_div = 200

    # --- Load x: stream (M_tiles, 1) = (1, 1), tile (B, N) ---
    x_load = OffChipLoad(underlying=x_2d, stride=(1, 0),
                         out_shape_tiled=(M_tiles, 1),
                         tile_row=tile_m, tile_col=tile_col,
                         par_dispatch=par_dispatch)

    # x is used for: (0) mean computation, (1) x - mean, (2) final multiply
    # But x - mean also feeds into variance, so we need:
    #   x -> (0) RowWiseSum for mean
    #   x -> (1) subtract mean (x_centered)
    # x_centered -> (0) variance computation, (1) final normalization
    x_bc = Broadcast(graph=graph, input=x_load, num_consumers=2)

    # --- Step 1: mean = RowWiseSum(x) / N ---
    sum_x = UnaryMap(graph=graph, input=(x_bc, 0),
                     fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=bw_div)
    n_load = _load_const(tile_m, M_tiles, float(N), par_dispatch)
    mean = BinaryMap(graph=graph, in1=sum_x, in2=n_load,
                     fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)

    # --- Step 2: x_centered = x + (-mean) ---
    neg_one_load = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    neg_mean = BinaryMap(graph=graph, in1=mean, in2=neg_one_load,
                         fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    # neg_mean is (tile_m, 1), x is (tile_m, N) — broadcasting works
    x_centered = BinaryMap(graph=graph, in1=(x_bc, 1), in2=neg_mean,
                           fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    # x_centered used for: (0) variance, (1) final normalization
    xc_bc = Broadcast(graph=graph, input=x_centered, num_consumers=2)

    # --- Step 3: var = RowWiseSum(x_centered^2) / N ---
    xc_sq = BinaryMap(graph=graph, in1=(xc_bc, 0), in2=(xc_bc, 0),
                      fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

    # Wait — a single broadcast output port can't be consumed twice by the same
    # BinaryMap. We need a separate broadcast for the squaring.
    # Let me restructure: xc_bc has 3 consumers instead.

    # Actually, let me re-do this. xc_bc needs: (0,1) for squaring, (2) for final.
    # But that means the broadcast needs 3 outputs. Let me fix.

    # Remove the bad nodes and redo
    graph.clear()  # start fresh — we'll rebuild

    graph = MultiDiGraph()

    x_load = OffChipLoad(underlying=x_2d, stride=(1, 0),
                         out_shape_tiled=(M_tiles, 1),
                         tile_row=tile_m, tile_col=tile_col,
                         par_dispatch=par_dispatch)

    x_bc = Broadcast(graph=graph, input=x_load, num_consumers=2)

    # mean = RowWiseSum(x) / N
    sum_x = UnaryMap(graph=graph, input=(x_bc, 0),
                     fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=bw_div)
    n_load = _load_const(tile_m, M_tiles, float(N), par_dispatch)
    mean = BinaryMap(graph=graph, in1=sum_x, in2=n_load,
                     fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)

    # neg_mean
    neg_one_load = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    neg_mean = BinaryMap(graph=graph, in1=mean, in2=neg_one_load,
                         fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

    # x_centered = x + neg_mean (broadcasts (tile_m,1) to (tile_m,N))
    x_centered = BinaryMap(graph=graph, in1=(x_bc, 1), in2=neg_mean,
                           fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    # x_centered -> (0,1) for squaring, (2) for final normalization
    xc_bc = Broadcast(graph=graph, input=x_centered, num_consumers=3)

    # var = RowWiseSum(x_centered^2) / N
    xc_sq = BinaryMap(graph=graph, in1=(xc_bc, 0), in2=(xc_bc, 1),
                      fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    sum_xc_sq = UnaryMap(graph=graph, input=xc_sq,
                         fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=bw_div)
    n_load2 = _load_const(tile_m, M_tiles, float(N), par_dispatch)
    var = BinaryMap(graph=graph, in1=sum_xc_sq, in2=n_load2,
                    fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)

    # var + eps
    eps_load = _load_const(tile_m, M_tiles, eps, par_dispatch)
    s = BinaryMap(graph=graph, in1=var, in2=eps_load,
                  fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)

    # --- rsqrt(s) via Pade + Newton-Raphson ---
    s_bc = Broadcast(graph=graph, input=s, num_consumers=NR_ITERS + 2)

    # y0 = exp(-(s-1)/(s+1))
    neg_ones_load = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    s_minus_1 = BinaryMap(graph=graph, in1=(s_bc, 0), in2=neg_ones_load,
                          fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    ones_load = _load_const(tile_m, M_tiles, 1.0, par_dispatch)
    s_plus_1 = BinaryMap(graph=graph, in1=(s_bc, 1), in2=ones_load,
                         fn=map_fn.Add(), write_back_mu=False, compute_bw=bw_add)
    ratio = BinaryMap(graph=graph, in1=s_minus_1, in2=s_plus_1,
                      fn=map_fn.Div(), write_back_mu=False, compute_bw=bw_div)
    neg_one_load2 = _load_const(tile_m, M_tiles, -1.0, par_dispatch)
    neg_ratio = BinaryMap(graph=graph, in1=neg_one_load2, in2=ratio,
                          fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)
    y = UnaryMap(graph=graph, input=neg_ratio,
                 fn=map_fn.Exp(), write_back_mu=False, compute_bw=bw_mul)

    # NR iterations: y = y * (1.5 + (-0.5)*s*y^2)
    for i in range(NR_ITERS):
        c15 = _load_const(tile_m, M_tiles, 1.5, par_dispatch)
        cn05 = _load_const(tile_m, M_tiles, -0.5, par_dispatch)
        y = _nr_rsqrt_iter(graph, y, (s_bc, i + 2), c15, cn05, bw_mul, bw_add)

    # --- normalized = x_centered * rsqrt ---
    normalized = BinaryMap(graph=graph, in1=(xc_bc, 2), in2=y,
                           fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

    # --- output = normalized * weight + bias ---
    weight_2d = weight.unsqueeze(0).expand(tile_m, N).contiguous()
    weight_load = OffChipLoad(underlying=weight_2d, stride=(0, 0),
                              out_shape_tiled=(M_tiles, 1),
                              tile_row=tile_m, tile_col=tile_col,
                              par_dispatch=par_dispatch)
    scaled = BinaryMap(graph=graph, in1=normalized, in2=weight_load,
                       fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul)

    bias_2d = bias.unsqueeze(0).expand(tile_m, N).contiguous()
    bias_load = OffChipLoad(underlying=bias_2d, stride=(0, 0),
                            out_shape_tiled=(M_tiles, 1),
                            tile_row=tile_m, tile_col=tile_col,
                            par_dispatch=par_dispatch)
    result = BinaryMap(graph=graph, in1=scaled, in2=bias_load,
                       fn=map_fn.Add(), write_back_mu=True, compute_bw=bw_add)

    output_op = OffChipStore(graph=graph, input=result,
                             par_dispatch=par_dispatch, store_file_name="output")

    graph = infer_broadcast(graph)
    return graph, output_op
