"""NKI baseline for gemm_scale_residual: Linear(x) * scaling_factor + Linear(x).

Computes x @ W^T + bias, then result * scaling_factor + result = result * (1 + scaling_factor).
"""

import numpy as np
import torch
import torch.nn as nn
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

SEED = 42
TILE_K = 128
TILE_M = 128
TILE_N = 128


class Model(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        original_x = x.clone().detach()
        x = x * self.scaling_factor
        x = x + original_x
        return x


def get_init_inputs(dims):
    return [dims["in_features"], dims["out_features"], dims["scaling_factor"]]


def get_nki_inputs(dims):
    # Create model to extract weights (consumes random state from seed)
    torch.manual_seed(SEED)
    model = Model(*get_init_inputs(dims))
    W = model.matmul.weight.detach().numpy()   # (out_features, in_features)
    b = model.matmul.bias.detach().numpy()      # (out_features,)
    W_T = np.ascontiguousarray(W.T)             # (in_features, out_features)
    b = b.reshape(1, -1)                         # (1, out_features) for 2D load

    # Generate inputs with fresh seed (matching get_inputs)
    torch.manual_seed(SEED)
    x = torch.randn(dims["batch_size"], dims["in_features"]).numpy()

    scaling_factor = np.float32(dims["scaling_factor"])
    return [x, W_T, b, scaling_factor]


@nki.jit
def nki_kernel(X, W_T, bias, scaling_factor):
    """Compute (X @ W_T + bias) * (1 + scaling_factor).

    X: (M, K), W_T: (K, N), bias: (1, N), scaling_factor: scalar.
    """
    M, K = X.shape
    _, N = W_T.shape

    tile_m = min(M, TILE_M)
    n_tiles_m = M // tile_m
    n_tiles_k = K // TILE_K
    n_tiles_n = N // TILE_N

    combined_factor = 1.0 + scaling_factor

    out = nl.ndarray((M, N), dtype=np.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(n_tiles_m):
        for n in nl.affine_range(n_tiles_n):
            acc = nl.zeros((tile_m, TILE_N), dtype=nl.float32, buffer=nl.psum)

            # Load bias tile for this N-tile: (1, TILE_N) for broadcast
            bias_tile = nl.load(
                bias[nl.arange(1)[:, None],
                     n * TILE_N + nl.arange(TILE_N)[None, :]],
                dtype=nl.float32)

            for k in nl.sequential_range(n_tiles_k):
                # X^T slice: (TILE_K, tile_m)
                x_tile = nl.load(
                    X[m * tile_m + nl.arange(tile_m)[:, None],
                      k * TILE_K + nl.arange(TILE_K)[None, :]]
                )
                x_t = nl.transpose(x_tile)

                w_tile = nl.load(
                    W_T[k * TILE_K + nl.arange(TILE_K)[:, None],
                        n * TILE_N + nl.arange(TILE_N)[None, :]]
                )
                acc += nisa.nc_matmul(x_t, w_tile)

            # Copy accumulator to sbuf for element-wise ops
            result = nl.copy(acc, dtype=nl.float32)

            # Add bias (broadcast across rows)
            result = nl.add(result, bias_tile)

            # Multiply by (1 + scaling_factor)
            result = nisa.tensor_scalar(data=result, op0=np.multiply,
                                        operand0=combined_factor)

            nl.store(
                out[m * tile_m + nl.arange(tile_m)[:, None],
                    n * TILE_N + nl.arange(TILE_N)[None, :]],
                value=result,
            )

    return out
