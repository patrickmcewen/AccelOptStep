"""NKI baseline for mlp_shallow_wide: shallow wide MLP with ReLU activations.

Same architecture as mlp but uses hidden_layer_sizes instead of layer_sizes.
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
    """Extract weights/biases from the PyTorch model, then generate input."""
    torch.manual_seed(SEED)
    from StepBench.problems.mlp_shallow_wide import Model, get_init_inputs
    model = Model(*get_init_inputs(dims))

    weights = []
    biases = []
    for module in model.network:
        if isinstance(module, nn.Linear):
            W = module.weight.detach().numpy()
            b = module.bias.detach().numpy()
            W_T = np.ascontiguousarray(W.T)
            b_2d = b.reshape(1, -1).copy()
            weights.append(W_T)
            biases.append(b_2d)

    torch.manual_seed(SEED)
    x = torch.randn(dims["batch_size"], dims["input_size"]).numpy()

    result = [x]
    for w, b in zip(weights, biases):
        result.append(w)
        result.append(b)
    return result


def _make_mlp_kernel(n_layers):
    """Build a 3-layer MLP NKI kernel (2 hidden + ReLU, 1 output)."""

    @nki.jit
    def _mlp_3layer(X, W0_T, b0, W1_T, b1, W2_T, b2):
        """3-layer MLP: Linear+ReLU, Linear+ReLU, Linear."""
        M = X.shape[0]
        tile_m = min(M, TILE_M)
        n_tiles_m = M // tile_m

        # Layer 0: X @ W0^T + b0, then ReLU
        K0 = W0_T.shape[0]
        N0 = W0_T.shape[1]
        buf0 = nl.ndarray((M, N0), dtype=np.float32, buffer=nl.shared_hbm)

        for m in nl.affine_range(n_tiles_m):
            for n in nl.affine_range(N0 // TILE_N):
                acc = nl.zeros((tile_m, TILE_N), dtype=nl.float32, buffer=nl.psum)
                bias_tile = nl.load(b0[nl.arange(1)[:, None],
                                       n * TILE_N + nl.arange(TILE_N)[None, :]],
                                    dtype=nl.float32)
                for k in nl.sequential_range(K0 // TILE_K):
                    x_tile = nl.load(X[m * tile_m + nl.arange(tile_m)[:, None],
                                       k * TILE_K + nl.arange(TILE_K)[None, :]])
                    x_t = nl.transpose(x_tile)
                    w_tile = nl.load(W0_T[k * TILE_K + nl.arange(TILE_K)[:, None],
                                          n * TILE_N + nl.arange(TILE_N)[None, :]])
                    acc += nisa.nc_matmul(x_t, w_tile)
                result = nl.copy(acc, dtype=nl.float32)
                result = nl.add(result, bias_tile)
                result = nisa.activation(op=nl.relu, data=result, dtype=np.float32)
                nl.store(buf0[m * tile_m + nl.arange(tile_m)[:, None],
                              n * TILE_N + nl.arange(TILE_N)[None, :]], value=result)

        # Layer 1: buf0 @ W1^T + b1, then ReLU
        K1 = W1_T.shape[0]
        N1 = W1_T.shape[1]
        buf1 = nl.ndarray((M, N1), dtype=np.float32, buffer=nl.shared_hbm)

        for m in nl.affine_range(n_tiles_m):
            for n in nl.affine_range(N1 // TILE_N):
                acc = nl.zeros((tile_m, TILE_N), dtype=nl.float32, buffer=nl.psum)
                bias_tile = nl.load(b1[nl.arange(1)[:, None],
                                       n * TILE_N + nl.arange(TILE_N)[None, :]],
                                    dtype=nl.float32)
                for k in nl.sequential_range(K1 // TILE_K):
                    x_tile = nl.load(buf0[m * tile_m + nl.arange(tile_m)[:, None],
                                          k * TILE_K + nl.arange(TILE_K)[None, :]])
                    x_t = nl.transpose(x_tile)
                    w_tile = nl.load(W1_T[k * TILE_K + nl.arange(TILE_K)[:, None],
                                          n * TILE_N + nl.arange(TILE_N)[None, :]])
                    acc += nisa.nc_matmul(x_t, w_tile)
                result = nl.copy(acc, dtype=nl.float32)
                result = nl.add(result, bias_tile)
                result = nisa.activation(op=nl.relu, data=result, dtype=np.float32)
                nl.store(buf1[m * tile_m + nl.arange(tile_m)[:, None],
                              n * TILE_N + nl.arange(TILE_N)[None, :]], value=result)

        # Layer 2: buf1 @ W2^T + b2 (no ReLU)
        K2 = W2_T.shape[0]
        N2 = W2_T.shape[1]
        out = nl.ndarray((M, N2), dtype=np.float32, buffer=nl.shared_hbm)

        for m in nl.affine_range(n_tiles_m):
            for n in nl.affine_range(N2 // TILE_N):
                acc = nl.zeros((tile_m, TILE_N), dtype=nl.float32, buffer=nl.psum)
                bias_tile = nl.load(b2[nl.arange(1)[:, None],
                                       n * TILE_N + nl.arange(TILE_N)[None, :]],
                                    dtype=nl.float32)
                for k in nl.sequential_range(K2 // TILE_K):
                    x_tile = nl.load(buf1[m * tile_m + nl.arange(tile_m)[:, None],
                                          k * TILE_K + nl.arange(TILE_K)[None, :]])
                    x_t = nl.transpose(x_tile)
                    w_tile = nl.load(W2_T[k * TILE_K + nl.arange(TILE_K)[:, None],
                                          n * TILE_N + nl.arange(TILE_N)[None, :]])
                    acc += nisa.nc_matmul(x_t, w_tile)
                result = nl.copy(acc, dtype=nl.float32)
                result = nl.add(result, bias_tile)
                nl.store(out[m * tile_m + nl.arange(tile_m)[:, None],
                             n * TILE_N + nl.arange(TILE_N)[None, :]], value=result)

        return out

    assert n_layers == 3, f"Only 3-layer MLP supported, got {n_layers}"
    return _mlp_3layer


def get_nki_kernel(dims):
    """Return a specialized NKI kernel for the given MLP dims."""
    hidden_layer_sizes = dims["hidden_layer_sizes"]
    n_layers = len(hidden_layer_sizes) + 1  # hidden layers + output layer
    return _make_mlp_kernel(n_layers)
