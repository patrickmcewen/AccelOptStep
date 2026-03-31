"""NKI baseline for layernorm: (x - mean) / sqrt(var + eps).

Input shape: (batch_size, features, dim1, dim2).
LayerNorm normalizes over the last 3 dims (features * dim1 * dim2).
With default weight=1, bias=0, this is just standardization.

Strategy: reshape to (B, N) where N = features*dim1*dim2. For each batch row,
compute mean and variance via E[x] and E[x^2], then normalize.
Uses two passes: first to compute statistics, second to normalize.
"""

import numpy as np
import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

SEED = 42
EPS = 1e-5
TILE_C = 512  # free-dim tile size


def get_nki_inputs(dims):
    torch.manual_seed(SEED)
    B, F, D1, D2 = dims["batch_size"], dims["features"], dims["dim1"], dims["dim2"]
    x = torch.randn(B, F, D1, D2).numpy()
    # Flatten to (B, F*D1*D2) — preserves (B, F, D1, D2) memory layout
    x_flat = x.reshape(B, -1)
    return [np.ascontiguousarray(x_flat)]


@nki.jit
def nki_kernel(x_flat):
    """LayerNorm: (x - mean) / sqrt(var + eps).

    x_flat: (B, N) where N = features * dim1 * dim2.
    Output: (B, N).
    """
    B, N = x_flat.shape

    n_tiles = N // TILE_C
    out = nl.ndarray((B, N), dtype=np.float32, buffer=nl.shared_hbm)

    inv_n = np.float32(1.0 / N)

    for b in nl.affine_range(B):
        # --- Pass 1: compute sum(x) and sum(x^2) across all tiles ---
        for t in nl.sequential_range(n_tiles):
            x_tile = nl.load(
                x_flat[b, t * TILE_C + nl.arange(TILE_C)[None, :]]
            )

            # Partial sums within this tile -> (1, 1)
            tile_sum = nisa.tensor_reduce(np.add, data=x_tile, axis=(1,))

            x_sq = nisa.tensor_tensor(x_tile, x_tile, op=np.multiply)
            tile_sum_sq = nisa.tensor_reduce(np.add, data=x_sq, axis=(1,))

            # Accumulate across tiles
            total_sum = nl.loop_reduce(tile_sum, op=np.add, loop_indices=[t])
            total_sum_sq = nl.loop_reduce(tile_sum_sq, op=np.add, loop_indices=[t])

        # mean = total_sum / N, mean_sq = total_sum_sq / N
        mean = nisa.tensor_scalar(data=total_sum, op0=np.multiply, operand0=inv_n)
        mean_sq = nisa.tensor_scalar(data=total_sum_sq, op0=np.multiply, operand0=inv_n)

        # var = E[x^2] - E[x]^2
        mean_squared = nisa.tensor_tensor(mean, mean, op=np.multiply)
        var = nisa.tensor_tensor(mean_sq, mean_squared, op=np.subtract)

        # var + eps
        var_eps = nisa.tensor_scalar(data=var, op0=np.add, operand0=np.float32(EPS))

        # rsqrt(var + eps)
        inv_std = nisa.activation(op=nl.rsqrt, data=var_eps)

        # --- Pass 2: normalize (x - mean) * inv_std ---
        for t2 in nl.affine_range(n_tiles):
            x_tile = nl.load(
                x_flat[b, t2 * TILE_C + nl.arange(TILE_C)[None, :]]
            )

            # x - mean: negate mean, then add
            neg_mean = nisa.tensor_scalar(data=mean, op0=np.multiply, operand0=np.float32(-1.0))
            x_centered = nisa.tensor_scalar(data=x_tile, op0=np.add, operand0=neg_mean)

            # (x - mean) * inv_std
            result = nisa.tensor_scalar(data=x_centered, op0=np.multiply, operand0=inv_std)

            nl.store(
                out[b, t2 * TILE_C + nl.arange(TILE_C)[None, :]],
                value=result,
            )

    return out
