"""NKI baseline for gemm_swish_scaling: Linear(x, bias=False) -> swish -> scale.

Computes x @ W^T, then result * sigmoid(result) * scaling_factor.
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
        self.matmul = nn.Linear(in_features, out_features, bias=False)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        x = x * torch.sigmoid(x)
        x = x * self.scaling_factor
        return x


def get_init_inputs(dims):
    return [dims["in_features"], dims["out_features"], dims["scaling_factor"]]


def get_nki_inputs(dims):
    # Create model to extract weights (consumes random state from seed)
    torch.manual_seed(SEED)
    model = Model(*get_init_inputs(dims))
    W = model.matmul.weight.detach().numpy()   # (out_features, in_features)
    W_T = np.ascontiguousarray(W.T)             # (in_features, out_features)

    # Generate inputs with fresh seed (matching get_inputs)
    torch.manual_seed(SEED)
    x = torch.randn(dims["batch_size"], dims["in_features"]).numpy()

    scaling_factor = np.float32(dims["scaling_factor"])
    return [x, W_T, scaling_factor]


@nki.jit
def nki_kernel(X, W_T, scaling_factor):
    """Compute (X @ W_T) * sigmoid(X @ W_T) * scaling_factor.

    X: (M, K), W_T: (K, N), scaling_factor: scalar.
    """
    M, K = X.shape
    _, N = W_T.shape

    tile_m = min(M, TILE_M)
    n_tiles_m = M // tile_m
    n_tiles_k = K // TILE_K
    n_tiles_n = N // TILE_N

    out = nl.ndarray((M, N), dtype=np.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(n_tiles_m):
        for n in nl.affine_range(n_tiles_n):
            acc = nl.zeros((tile_m, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for k in nl.sequential_range(n_tiles_k):
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

            # Copy accumulator to sbuf
            matmul_result = nl.copy(acc, dtype=nl.float32)

            # Swish: x * sigmoid(x)
            sig = nisa.activation(op=nl.sigmoid, data=matmul_result)
            swish = nisa.tensor_tensor(matmul_result, sig, op=np.multiply)

            # Scale
            result = nisa.tensor_scalar(data=swish, op0=np.multiply,
                                        operand0=scaling_factor)

            nl.store(
                out[m * tile_m + nl.arange(tile_m)[:, None],
                    n * TILE_N + nl.arange(TILE_N)[None, :]],
                value=result,
            )

    return out
