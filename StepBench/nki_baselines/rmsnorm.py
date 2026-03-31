"""NKI baseline for rmsnorm: output = x / sqrt(mean(x^2, dim=1) + eps).

Input shape: (batch_size, features, dim1, dim2).
Reduction over dim=1 (features).

Strategy: pass data as (B, D1*D2, F) so reduction is over free dim (axis 1 = F).
For each (batch, spatial_tile), load (TILE_P, F), reduce, normalize, transpose
to (F, TILE_P), and store back to (B, F, D1*D2) output to match gold layout.
"""

import numpy as np
import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

SEED = 42
EPS = 1e-5
TILE_P = 128  # par_dim tile size (spatial positions per tile)


def get_nki_inputs(dims):
    torch.manual_seed(SEED)
    B, F, D1, D2 = dims["batch_size"], dims["features"], dims["dim1"], dims["dim2"]
    x = torch.randn(B, F, D1, D2).numpy()
    # Reshape to (B, D1*D2, F): transpose features to free dim
    x_in = np.ascontiguousarray(x.reshape(B, F, D1 * D2).transpose(0, 2, 1))
    return [x_in]


@nki.jit
def nki_kernel(x_in):
    """RMSNorm: x / sqrt(mean(x^2, dim=-1) + eps), then transpose back.

    x_in: (B, S, F) where S = dim1*dim2, F = features.
    Output: (B, F, S) to match gold layout.
    """
    B, S, F = x_in.shape

    n_tiles_s = S // TILE_P
    out = nl.ndarray((B, F, S), dtype=np.float32, buffer=nl.shared_hbm)

    inv_f = np.float32(1.0 / F)

    for b in nl.affine_range(B):
        for t in nl.affine_range(n_tiles_s):
            s_off = t * TILE_P

            # Load tile: (TILE_P, F) — par_dim is spatial, free dim is features
            x_tile = nl.load(
                x_in[b, s_off + nl.arange(TILE_P)[:, None],
                     nl.arange(F)[None, :]]
            )

            # x^2
            x_sq = nisa.tensor_tensor(x_tile, x_tile, op=np.multiply)

            # sum(x^2) over features (free dim, axis 1) -> (TILE_P, 1)
            sum_sq = nisa.tensor_reduce(np.add, data=x_sq, axis=(1,))

            # mean = sum / F
            mean_sq = nisa.tensor_scalar(data=sum_sq, op0=np.multiply, operand0=inv_f)

            # mean + eps
            mean_eps = nisa.tensor_scalar(data=mean_sq, op0=np.add, operand0=np.float32(EPS))

            # rsqrt(mean + eps) -> (TILE_P, 1)
            inv_rms = nisa.activation(op=nl.rsqrt, data=mean_eps)

            # x * inv_rms: broadcast (TILE_P, 1) across F columns -> (TILE_P, F)
            result = nisa.tensor_scalar(data=x_tile, op0=np.multiply, operand0=inv_rms)

            # Transpose (TILE_P, F) -> (F, TILE_P) and store to (B, F, S) output
            result_t = nl.transpose(result)

            nl.store(
                out[b, nl.arange(F)[:, None],
                    s_off + nl.arange(TILE_P)[None, :]],
                value=result_t,
            )

    return out
