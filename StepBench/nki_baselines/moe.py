"""NKI baseline for moe: Mixture of Experts with top-1 routing.

Pre-computes routing masks on CPU. Computes all expert matmuls for all tokens,
then combines using masks. Supports 2 experts (small preset).
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


def get_nki_inputs(dims):
    """Extract weights and pre-compute routing masks."""
    torch.manual_seed(SEED)
    from StepBench.problems.moe import Model, get_init_inputs
    model = Model(*get_init_inputs(dims))

    gate_W = model.gate.weight.detach().numpy()          # (num_experts, hidden_dim)
    expert_Ws = [e.weight.detach().numpy()                # (intermediate_dim, hidden_dim)
                 for e in model.experts]

    # Fresh seed for input
    torch.manual_seed(SEED)
    x_np = torch.randn(dims["num_tokens"], dims["hidden_dim"]).numpy()

    # Pre-compute routing: gate_scores = x @ gate_W^T
    gate_scores = x_np @ gate_W.T                         # (num_tokens, num_experts)
    expert_idx = np.argmax(gate_scores, axis=-1)           # (num_tokens,)

    num_experts = dims["num_experts"]
    num_tokens = dims["num_tokens"]
    intermediate_dim = dims["intermediate_dim"]

    # Build masks: (num_tokens, 1) per expert, broadcast over intermediate_dim
    masks = []
    for i in range(num_experts):
        mask = (expert_idx == i).astype(np.float32).reshape(num_tokens, 1)
        # Expand to (num_tokens, intermediate_dim) for element-wise multiply
        mask_full = np.broadcast_to(mask, (num_tokens, intermediate_dim)).copy()
        masks.append(mask_full)

    # Transpose expert weights for matmul: W_T = (hidden_dim, intermediate_dim)
    expert_Ws_T = [np.ascontiguousarray(w.T) for w in expert_Ws]

    # Return: x, W0_T, W1_T, mask0, mask1
    result = [x_np]
    for w in expert_Ws_T:
        result.append(w)
    for m in masks:
        result.append(m)
    return result


def _make_moe_kernel_2experts():
    """NKI kernel for 2-expert MoE."""

    @nki.jit
    def _moe_2(X, W0_T, W1_T, mask0, mask1):
        """MoE with 2 experts. X: (M,K), Wi_T: (K,N), maski: (M,N)."""
        M, K = X.shape
        _, N = W0_T.shape

        tile_m = min(M, TILE_M)
        tile_k = min(K, TILE_K)
        tile_n = min(N, TILE_N)
        n_tiles_m = M // tile_m
        n_tiles_k = K // tile_k
        n_tiles_n = N // tile_n

        out = nl.ndarray((M, N), dtype=np.float32, buffer=nl.shared_hbm)

        for m in nl.affine_range(n_tiles_m):
            for n in nl.affine_range(n_tiles_n):
                # Expert 0 matmul
                acc0 = nl.zeros((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
                for k in nl.sequential_range(n_tiles_k):
                    x_tile = nl.load(X[m * tile_m + nl.arange(tile_m)[:, None],
                                       k * tile_k + nl.arange(tile_k)[None, :]])
                    x_t = nl.transpose(x_tile)
                    w_tile = nl.load(W0_T[k * tile_k + nl.arange(tile_k)[:, None],
                                          n * tile_n + nl.arange(tile_n)[None, :]])
                    acc0 += nisa.nc_matmul(x_t, w_tile)
                res0 = nl.copy(acc0, dtype=nl.float32)

                # Expert 1 matmul
                acc1 = nl.zeros((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
                for k in nl.sequential_range(n_tiles_k):
                    x_tile = nl.load(X[m * tile_m + nl.arange(tile_m)[:, None],
                                       k * tile_k + nl.arange(tile_k)[None, :]])
                    x_t = nl.transpose(x_tile)
                    w_tile = nl.load(W1_T[k * tile_k + nl.arange(tile_k)[:, None],
                                          n * tile_n + nl.arange(tile_n)[None, :]])
                    acc1 += nisa.nc_matmul(x_t, w_tile)
                res1 = nl.copy(acc1, dtype=nl.float32)

                # Load masks and combine
                m0 = nl.load(mask0[m * tile_m + nl.arange(tile_m)[:, None],
                                    n * tile_n + nl.arange(tile_n)[None, :]],
                             dtype=nl.float32)
                m1 = nl.load(mask1[m * tile_m + nl.arange(tile_m)[:, None],
                                    n * tile_n + nl.arange(tile_n)[None, :]],
                             dtype=nl.float32)

                # output = mask0 * expert0_out + mask1 * expert1_out
                out0 = nl.multiply(res0, m0, dtype=np.float32)
                out1 = nl.multiply(res1, m1, dtype=np.float32)
                combined = nl.add(out0, out1, dtype=np.float32)

                nl.store(out[m * tile_m + nl.arange(tile_m)[:, None],
                             n * tile_n + nl.arange(tile_n)[None, :]], value=combined)

        return out

    return _moe_2


def get_nki_kernel(dims):
    """Return a specialized NKI kernel for the given MoE dims."""
    num_experts = dims["num_experts"]
    assert num_experts == 2, f"Only 2-expert MoE supported, got {num_experts}"
    return _make_moe_kernel_2experts()
