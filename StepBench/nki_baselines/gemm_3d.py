"""NKI baseline for gemm_3d: C = A @ B, shapes (N,M,K) @ (K,L) -> (N,M,L)."""

import numpy as np
import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

SEED = 42
TILE_K = 128
TILE_M = 128
TILE_L = 64


def compute_gold(dims):
    torch.manual_seed(SEED)
    N, M, K, L = dims["N"], dims["M"], dims["K"], dims["L"]
    A = torch.randn(N, M, K)
    B = torch.randn(K, L)
    return torch.matmul(A, B)


def get_nki_inputs(dims):
    torch.manual_seed(SEED)
    N, M, K, L = dims["N"], dims["M"], dims["K"], dims["L"]
    A = torch.randn(N, M, K).numpy()
    B = torch.randn(K, L).numpy()
    # Transpose A along last two dims: (N, M, K) -> (N, K, M)
    A_T = np.ascontiguousarray(A.transpose(0, 2, 1))
    return [A_T, B]


@nki.jit
def nki_kernel(A_T, B):
    """Compute A @ B for each batch. A_T is (N, K, M), B is (K, L)."""
    N_batch, K, M = A_T.shape
    _, L = B.shape

    tile_m = min(M, TILE_M)
    n_tiles_m = M // tile_m
    n_tiles_k = K // TILE_K
    n_tiles_l = L // TILE_L

    out = nl.ndarray((N_batch, M, L), dtype=np.float32, buffer=nl.shared_hbm)

    for n in nl.affine_range(N_batch):
        for m in nl.affine_range(n_tiles_m):
            for l in nl.affine_range(n_tiles_l):
                acc = nl.zeros((tile_m, TILE_L), dtype=nl.float32, buffer=nl.psum)
                for k in nl.sequential_range(n_tiles_k):
                    lhs = nl.load(
                        A_T[n,
                            k * TILE_K + nl.arange(TILE_K)[:, None],
                            m * tile_m + nl.arange(tile_m)[None, :]]
                    )
                    rhs = nl.load(
                        B[k * TILE_K + nl.arange(TILE_K)[:, None],
                          l * TILE_L + nl.arange(TILE_L)[None, :]]
                    )
                    acc += nisa.nc_matmul(lhs, rhs)

                nl.store(
                    out[n,
                        m * tile_m + nl.arange(tile_m)[:, None],
                        l * TILE_L + nl.arange(TILE_L)[None, :]],
                    value=acc,
                )

    return out
