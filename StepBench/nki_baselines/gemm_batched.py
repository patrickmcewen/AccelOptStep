"""NKI baseline for gemm_batched: C = bmm(A, B), shapes (B,M,K) @ (B,K,N) -> (B,M,N)."""

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
    batch, M, K, N = dims["batch"], dims["M"], dims["K"], dims["N"]
    A = torch.randn(batch, M, K)
    B = torch.randn(batch, K, N)
    return torch.bmm(A, B)


def get_nki_inputs(dims):
    torch.manual_seed(SEED)
    batch, M, K, N = dims["batch"], dims["M"], dims["K"], dims["N"]
    A = torch.randn(batch, M, K).numpy()
    B = torch.randn(batch, K, N).numpy()
    # Transpose A along last two dims: (batch, M, K) -> (batch, K, M)
    A_T = np.ascontiguousarray(A.transpose(0, 2, 1))
    return [A_T, B]


@nki.jit
def nki_kernel(A_T, B):
    """Compute batched A @ B. A_T is (batch, K, M), B is (batch, K, N)."""
    batch, K, M = A_T.shape
    _, _, N = B.shape

    tile_m = min(M, TILE_M)
    n_tiles_m = M // tile_m
    n_tiles_k = K // TILE_K
    n_tiles_n = N // TILE_N

    out = nl.ndarray((batch, M, N), dtype=np.float32, buffer=nl.shared_hbm)

    for b in nl.affine_range(batch):
        for m in nl.affine_range(n_tiles_m):
            for n in nl.affine_range(n_tiles_n):
                acc = nl.zeros((tile_m, TILE_N), dtype=nl.float32, buffer=nl.psum)
                for k in nl.sequential_range(n_tiles_k):
                    lhs = nl.load(
                        A_T[b,
                            k * TILE_K + nl.arange(TILE_K)[:, None],
                            m * tile_m + nl.arange(tile_m)[None, :]]
                    )
                    rhs = nl.load(
                        B[b,
                          k * TILE_K + nl.arange(TILE_K)[:, None],
                          n * TILE_N + nl.arange(TILE_N)[None, :]]
                    )
                    acc += nisa.nc_matmul(lhs, rhs)

                nl.store(
                    out[b,
                        m * tile_m + nl.arange(tile_m)[:, None],
                        n * TILE_N + nl.arange(TILE_N)[None, :]],
                    value=acc,
                )

    return out
