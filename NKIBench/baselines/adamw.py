"""adamw: AdamW optimizer step

theta_t = theta * (1 - weight_decay)   where weight_decay = 1e-5
m_t = beta1 * m + (1 - beta1) * g      where beta1 = 0.9
v_t = beta2 * v + (1 - beta2) * g^2    where beta2 = 0.999
v_hat = v_t / (1 - beta2)              = v_t * 1000
new_theta = theta_t - lr * m_t / (sqrt(v_hat) + eps)

For sqrt: Babylonian method (Heron's method) which converges for any
positive starting value. x_{n+1} = 0.5 * (x_n + a / x_n).
"""
import numpy as np
import torch
from networkx import MultiDiGraph

from step_py.functions import map_fn
from step_py.ops import OffChipLoad, BinaryMap, Broadcast, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42
SQRT_ITERS = 10


def compute_gold(dims):
    M, N = dims["M"], dims["N"]
    np.random.seed(SEED)
    theta = np.random.normal(0, 1.0, (M, N)).astype(np.float32)
    g = np.random.normal(0, 1.0, (M, N)).astype(np.float32)
    m = np.random.normal(0, 1.0, (M, N)).astype(np.float32)
    v = np.abs(np.random.normal(0, 1.0, (M, N))).astype(np.float32)

    theta_t = theta - 1e-5 * theta
    m_t = 0.9 * m + 0.1 * g
    v_t = 0.999 * v + 0.001 * g * g
    v_hat = v_t * 1000
    new_theta_t = theta_t - 0.01 * m_t / (np.sqrt(v_hat) + 1e-8)
    return torch.from_numpy(new_theta_t).float()


def _load_const(tile_m, val, M_tiles, N_tiles, par_dispatch):
    t = torch.full((tile_m, 1), val, dtype=torch.float32)
    return OffChipLoad(
        underlying=t, stride=(0, 0),
        out_shape_tiled=(M_tiles, N_tiles),
        tile_row=tile_m, tile_col=1, par_dispatch=par_dispatch,
    )


def _babylonian_sqrt_iter(graph, x_prev, a, half_const, bw):
    """One Babylonian iteration: x_new = 0.5 * (x + a/x)."""
    x_bc = Broadcast(graph=graph, input=x_prev, num_consumers=2)
    quotient = BinaryMap(graph=graph, in1=a, in2=(x_bc, 0),
                         fn=map_fn.Div(), write_back_mu=False, compute_bw=bw)
    x_plus_q = BinaryMap(graph=graph, in1=(x_bc, 1), in2=quotient,
                         fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)
    x_new = BinaryMap(graph=graph, in1=x_plus_q, in2=half_const,
                      fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)
    return x_new


def build_graph(dims):
    M, N = dims["M"], dims["N"]
    tile_m, tile_n = 64, 64
    par_dispatch = 4
    bw = 200

    M_tiles = M // tile_m
    N_tiles = N // tile_n

    graph = MultiDiGraph()

    np.random.seed(SEED)
    theta_np = np.random.normal(0, 1.0, (M, N)).astype(np.float32)
    g_np = np.random.normal(0, 1.0, (M, N)).astype(np.float32)
    m_np = np.random.normal(0, 1.0, (M, N)).astype(np.float32)
    v_np = np.abs(np.random.normal(0, 1.0, (M, N))).astype(np.float32)

    theta_t = torch.from_numpy(theta_np)
    g_t = torch.from_numpy(g_np)
    m_t = torch.from_numpy(m_np)
    v_t = torch.from_numpy(v_np)

    def _load_tensor(t):
        return OffChipLoad(
            underlying=t, stride=(N_tiles, 1),
            out_shape_tiled=(M_tiles, N_tiles),
            tile_row=tile_m, tile_col=tile_n, par_dispatch=par_dispatch,
        )

    def _const(val):
        return _load_const(tile_m, val, M_tiles, N_tiles, par_dispatch)

    # Load inputs
    theta_load = _load_tensor(theta_t)
    g_load = _load_tensor(g_t)
    m_load = _load_tensor(m_t)
    v_load = _load_tensor(v_t)

    # g is used for: (0) m_t, (1,2) g^2
    g_bc = Broadcast(graph=graph, input=g_load, num_consumers=3)

    # theta_t = theta * 0.99999
    wd_const = _const(1.0 - 1e-5)
    theta_decayed = BinaryMap(
        graph=graph, in1=theta_load, in2=wd_const,
        fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw,
    )

    # m_t = 0.9 * m + 0.1 * g
    m_scaled = BinaryMap(graph=graph, in1=m_load, in2=_const(0.9),
                         fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)
    g_scaled = BinaryMap(graph=graph, in1=(g_bc, 0), in2=_const(0.1),
                         fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)
    m_new = BinaryMap(graph=graph, in1=m_scaled, in2=g_scaled,
                      fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)

    # g^2
    g_sq = BinaryMap(graph=graph, in1=(g_bc, 1), in2=(g_bc, 2),
                     fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)

    # v_t = 0.999 * v + 0.001 * g^2
    v_scaled = BinaryMap(graph=graph, in1=v_load, in2=_const(0.999),
                         fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)
    g_sq_scaled = BinaryMap(graph=graph, in1=g_sq, in2=_const(0.001),
                            fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)
    v_new = BinaryMap(graph=graph, in1=v_scaled, in2=g_sq_scaled,
                      fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)

    # v_hat = v_new * 1000
    v_hat = BinaryMap(graph=graph, in1=v_new, in2=_const(1000.0),
                      fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)

    # v_hat_safe = v_hat + small_eps to avoid sqrt(0) -> 0/0 in Babylonian
    v_hat_safe = BinaryMap(graph=graph, in1=v_hat, in2=_const(1e-12),
                           fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)

    # sqrt(v_hat) via Babylonian method, starting from x=1
    # v_hat_safe is broadcast to all iterations
    a_bc = Broadcast(graph=graph, input=v_hat_safe, num_consumers=SQRT_ITERS)

    x = _const(1.0)  # initial estimate
    for i in range(SQRT_ITERS):
        half = _const(0.5)
        x = _babylonian_sqrt_iter(graph, x, (a_bc, i), half, bw)

    # denom = sqrt(v_hat) + 1e-8
    denom = BinaryMap(graph=graph, in1=x, in2=_const(1e-8),
                      fn=map_fn.Add(), write_back_mu=False, compute_bw=bw)

    # frac = m_t / denom
    frac = BinaryMap(graph=graph, in1=m_new, in2=denom,
                     fn=map_fn.Div(), write_back_mu=False, compute_bw=bw)

    # neg_update = frac * (-lr)
    neg_update = BinaryMap(graph=graph, in1=frac, in2=_const(-0.01),
                           fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw)

    # result = theta_t + neg_update
    result = BinaryMap(graph=graph, in1=theta_decayed, in2=neg_update,
                       fn=map_fn.Add(), write_back_mu=True, compute_bw=bw)

    output_op = OffChipStore(
        graph=graph, input=result,
        par_dispatch=par_dispatch, store_file_name="output",
    )
    graph = infer_broadcast(graph)
    return graph, output_op
