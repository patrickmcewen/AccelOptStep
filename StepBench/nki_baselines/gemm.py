"""NKI baseline for gemm: C = A @ B, shapes (M,K) @ (K,N) -> (M,N)."""

import numpy as np
import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

SEED = 42
TILE_K = 128
TILE_M = 128
TILE_N = 128


def compute_gold(dims):
    torch.manual_seed(SEED)
    M, K, N = dims["M"], dims["K"], dims["N"]
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return torch.matmul(A, B)


def get_nki_inputs(dims):
    torch.manual_seed(SEED)
    M, K, N = dims["M"], dims["K"], dims["N"]
    A = torch.randn(M, K).numpy()
    B = torch.randn(K, N).numpy()
    # Transpose A so kernel can use nc_matmul(A_T_tile, B_tile) = A_tile @ B_tile
    A_T = np.ascontiguousarray(A.T)
    return [A_T, B]


@nki.jit
def nki_kernel(A_T, B):
    """Compute A @ B. A_T is (K, M) = transpose of original A. B is (K, N)."""
    K, M = A_T.shape
    _, N = B.shape

    tile_m = min(M, TILE_M)
    n_tiles_m = M // tile_m
    n_tiles_k = K // TILE_K
    n_tiles_n = N // TILE_N

    out = nl.ndarray((M, N), dtype=np.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(n_tiles_m):
        for n in nl.affine_range(n_tiles_n):
            acc = nl.zeros((tile_m, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k in nl.sequential_range(n_tiles_k):
                lhs = nl.load(
                    A_T[k * TILE_K + nl.arange(TILE_K)[:, None],
                        m * tile_m + nl.arange(tile_m)[None, :]]
                )
                rhs = nl.load(
                    B[k * TILE_K + nl.arange(TILE_K)[:, None],
                      n * TILE_N + nl.arange(TILE_N)[None, :]]
                )
                acc += nisa.nc_matmul(lhs, rhs)

            nl.store(
                out[m * tile_m + nl.arange(tile_m)[:, None],
                    n * TILE_N + nl.arange(TILE_N)[None, :]],
                value=acc,
            )

    return out
