"""NKI baseline for activation: supports relu, sigmoid, swish, gelu, softmax."""

import numpy as np
import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

SEED = 42
TILE_P = 128   # partition dimension (rows)
TILE_F = 512   # free dimension (columns per chunk)


def get_nki_inputs(dims):
    torch.manual_seed(SEED)
    x = torch.randn(dims["batch_size"], dims["dim"]).numpy()
    return [x]


def get_nki_kernel(dims):
    """Return the appropriate JIT kernel for the activation function."""
    fn = dims["fn"]
    if fn == "softmax":
        return _get_softmax_kernel(dims["dim"])
    dispatch = {
        "relu": nki_relu,
        "sigmoid": nki_sigmoid,
        "swish": nki_swish,
        "gelu": nki_gelu,
    }
    return dispatch[fn]


# ---------------------------------------------------------------------------
# relu: max(0, x)
# ---------------------------------------------------------------------------
@nki.jit
def nki_relu(x):
    batch, dim = x.shape
    out = nl.ndarray((batch, dim), dtype=np.float32, buffer=nl.shared_hbm)
    n_p = batch // TILE_P
    n_f = dim // TILE_F

    for p in nl.affine_range(n_p):
        for f in nl.affine_range(n_f):
            tile = nl.load(x[p * TILE_P + nl.arange(TILE_P)[:, None],
                             f * TILE_F + nl.arange(TILE_F)[None, :]])
            result = nisa.activation(op=nl.relu, data=tile, dtype=np.float32)
            nl.store(out[p * TILE_P + nl.arange(TILE_P)[:, None],
                         f * TILE_F + nl.arange(TILE_F)[None, :]], value=result)
    return out


# ---------------------------------------------------------------------------
# sigmoid: 1 / (1 + exp(-x))
# ---------------------------------------------------------------------------
@nki.jit
def nki_sigmoid(x):
    batch, dim = x.shape
    out = nl.ndarray((batch, dim), dtype=np.float32, buffer=nl.shared_hbm)
    n_p = batch // TILE_P
    n_f = dim // TILE_F

    for p in nl.affine_range(n_p):
        for f in nl.affine_range(n_f):
            tile = nl.load(x[p * TILE_P + nl.arange(TILE_P)[:, None],
                             f * TILE_F + nl.arange(TILE_F)[None, :]])
            result = nisa.activation(op=nl.sigmoid, data=tile, dtype=np.float32)
            nl.store(out[p * TILE_P + nl.arange(TILE_P)[:, None],
                         f * TILE_F + nl.arange(TILE_F)[None, :]], value=result)
    return out


# ---------------------------------------------------------------------------
# swish: x * sigmoid(x)
# ---------------------------------------------------------------------------
@nki.jit
def nki_swish(x):
    batch, dim = x.shape
    out = nl.ndarray((batch, dim), dtype=np.float32, buffer=nl.shared_hbm)
    n_p = batch // TILE_P
    n_f = dim // TILE_F

    for p in nl.affine_range(n_p):
        for f in nl.affine_range(n_f):
            tile = nl.load(x[p * TILE_P + nl.arange(TILE_P)[:, None],
                             f * TILE_F + nl.arange(TILE_F)[None, :]])
            sig = nisa.activation(op=nl.sigmoid, data=tile, dtype=np.float32)
            result = nl.multiply(tile, sig, dtype=np.float32)
            nl.store(out[p * TILE_P + nl.arange(TILE_P)[:, None],
                         f * TILE_F + nl.arange(TILE_F)[None, :]], value=result)
    return out


# ---------------------------------------------------------------------------
# gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
# Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# tanh(z) = 2*sigmoid(2z) - 1, so:
# gelu(x) = 0.5 * x * (2 * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 * x^3)))
#         = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 * x^3))
# ---------------------------------------------------------------------------
GELU_COEFF = np.float32(2.0 * np.sqrt(2.0 / np.pi))  # ~1.5957691
GELU_CUBIC = np.float32(0.044715)


@nki.jit
def nki_gelu(x):
    batch, dim = x.shape
    out = nl.ndarray((batch, dim), dtype=np.float32, buffer=nl.shared_hbm)
    n_p = batch // TILE_P
    n_f = dim // TILE_F

    for p in nl.affine_range(n_p):
        for f in nl.affine_range(n_f):
            tile = nl.load(x[p * TILE_P + nl.arange(TILE_P)[:, None],
                             f * TILE_F + nl.arange(TILE_F)[None, :]])
            # x^2
            x2 = nl.multiply(tile, tile, dtype=np.float32)
            # 0.044715 * x^2
            cubic_term = nisa.tensor_scalar(data=x2, op0=np.multiply,
                                            operand0=GELU_CUBIC,
                                            dtype=np.float32)
            # 1 + 0.044715 * x^2
            inner = nisa.tensor_scalar(data=cubic_term, op0=np.add,
                                       operand0=np.float32(1.0),
                                       dtype=np.float32)
            # x * (1 + 0.044715 * x^2) = x + 0.044715 * x^3
            inner = nl.multiply(tile, inner, dtype=np.float32)
            # GELU_COEFF * (x + 0.044715 * x^3)
            z = nisa.tensor_scalar(data=inner, op0=np.multiply,
                                   operand0=GELU_COEFF,
                                   dtype=np.float32)
            # sigmoid(z) -- this gives us 0.5*(1+tanh(z/2)), but we used 2*sqrt(2/pi)
            sig = nisa.activation(op=nl.sigmoid, data=z, dtype=np.float32)
            # x * sigmoid(z)
            result = nl.multiply(tile, sig, dtype=np.float32)
            nl.store(out[p * TILE_P + nl.arange(TILE_P)[:, None],
                         f * TILE_F + nl.arange(TILE_F)[None, :]], value=result)
    return out


# ---------------------------------------------------------------------------
# softmax: exp(x - max(x, dim=1)) / sum(exp(x - max(x, dim=1)), dim=1)
# Requires full row access for reduction, so tile only along batch dim.
# ---------------------------------------------------------------------------
def _make_softmax_kernel(tile_f):
    """Create a softmax JIT kernel with the given free-dim tile size."""
    @nki.jit
    def _softmax(x):
        batch, dim = x.shape
        out = nl.ndarray((batch, dim), dtype=np.float32, buffer=nl.shared_hbm)
        n_p = batch // TILE_P
        n_f = dim // tile_f

        p_idx = nl.arange(TILE_P)[:, None]
        f_idx = nl.arange(tile_f)[None, :]

        for p in nl.affine_range(n_p):
            # Allocate accumulators outside loops so they survive scope
            row_max = nl.ndarray((TILE_P, 1), dtype=np.float32, buffer=nl.sbuf)
            row_sum = nl.ndarray((TILE_P, 1), dtype=np.float32, buffer=nl.sbuf)
            row_max[p_idx, 0] = nl.full((TILE_P, 1), fill_value=-np.float32(np.inf),
                                        dtype=np.float32, buffer=nl.sbuf)
            row_sum[p_idx, 0] = nl.zeros((TILE_P, 1), dtype=np.float32, buffer=nl.sbuf)

            # Phase 1: row-wise max across all column tiles
            for f in nl.sequential_range(n_f):
                tile = nl.load(x[p * TILE_P + p_idx, f * tile_f + f_idx])
                tile_max = nisa.tensor_reduce(data=tile, op=np.max, axis=(1,),
                                              dtype=np.float32)
                row_max[p_idx, 0] = nl.maximum(row_max, tile_max, dtype=np.float32)

            # Phase 2: exp(x - max) and row-wise sum
            for f in nl.sequential_range(n_f):
                tile = nl.load(x[p * TILE_P + p_idx, f * tile_f + f_idx])
                shifted = nl.subtract(tile, row_max, dtype=np.float32)
                exp_tile = nisa.activation(op=nl.exp, data=shifted, dtype=np.float32)
                tile_sum = nisa.tensor_reduce(data=exp_tile, op=np.add, axis=(1,),
                                              dtype=np.float32)
                row_sum[p_idx, 0] = nl.add(row_sum, tile_sum, dtype=np.float32)

            # Phase 3: divide exp(x - max) by sum
            inv_sum = nisa.reciprocal(data=row_sum, dtype=np.float32)
            for f in nl.affine_range(n_f):
                tile = nl.load(x[p * TILE_P + p_idx, f * tile_f + f_idx])
                shifted = nl.subtract(tile, row_max, dtype=np.float32)
                exp_tile = nisa.activation(op=nl.exp, data=shifted, dtype=np.float32)
                result = nl.multiply(exp_tile, inv_sum, dtype=np.float32)
                nl.store(out[p * TILE_P + p_idx, f * tile_f + f_idx], value=result)
        return out
    return _softmax


# Cache softmax kernels by tile size
_softmax_cache = {}


def _get_softmax_kernel(dim):
    """Return a softmax JIT kernel with tile_f chosen to evenly divide dim."""
    tile_f = min(dim, TILE_F)
    assert dim % tile_f == 0, f"dim={dim} not divisible by tile_f={tile_f}"
    if tile_f not in _softmax_cache:
        _softmax_cache[tile_f] = _make_softmax_kernel(tile_f)
    return _softmax_cache[tile_f]
