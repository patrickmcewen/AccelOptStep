"""NKI baseline for mlp: multi-layer perceptron with ReLU activations.

Computes Sequential(Linear+ReLU, ..., Linear) matching the PyTorch Model.
Uses get_nki_kernel(dims) factory to specialize for the layer configuration.
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
    from StepBench.problems.mlp import Model, get_init_inputs
    model = Model(*get_init_inputs(dims))

    weights = []
    biases = []
    for module in model.network:
        if isinstance(module, nn.Linear):
            W = module.weight.detach().numpy()        # (out, in)
            b = module.bias.detach().numpy()           # (out,)
            W_T = np.ascontiguousarray(W.T)            # (in, out)
            b_2d = b.reshape(1, -1).copy()             # (1, out)
            weights.append(W_T)
            biases.append(b_2d)

    # Fresh seed for input (matches get_inputs)
    torch.manual_seed(SEED)
    x = torch.randn(dims["batch_size"], dims["input_size"]).numpy()

    # Return: x, W0_T, b0, W1_T, b1, ..., Wn_T, bn
    result = [x]
    for w, b in zip(weights, biases):
        result.append(w)
        result.append(b)
    return result


def _make_mlp_kernel(n_layers, layer_in_sizes, layer_out_sizes, batch_size, has_relu):
    """Build an NKI kernel for a specific MLP architecture.

    n_layers: number of Linear layers
    layer_in_sizes: list of input sizes per layer
    layer_out_sizes: list of output sizes per layer
    batch_size: batch dimension
    has_relu: list of bools, whether each layer is followed by ReLU
    """

    @nki.jit
    def _mlp_2layer_relu(X, W0_T, b0, W1_T, b1, W2_T, b2):
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

    assert n_layers == 3, f"Only 3-layer MLP supported (2 hidden + 1 output), got {n_layers}"
    return _mlp_2layer_relu


def get_nki_kernel(dims):
    """Return a specialized NKI kernel for the given MLP dims."""
    layer_sizes = dims["layer_sizes"]
    input_size = dims["input_size"]
    output_size = dims["output_size"]
    batch_size = dims["batch_size"]

    # Build layer specs: [input->hidden0+relu, hidden0->hidden1+relu, ..., hiddenN->output]
    sizes_in = [input_size] + list(layer_sizes)
    sizes_out = list(layer_sizes) + [output_size]
    n_layers = len(sizes_in)
    has_relu = [True] * len(layer_sizes) + [False]

    return _make_mlp_kernel(n_layers, sizes_in, sizes_out, batch_size, has_relu)
