"""NKI baseline for sdpa: scaled_dot_product_attention(Q, K, V).

Algorithm:
  scores = Q @ K^T / sqrt(dim)      -> (batch, heads, seq, seq)
  attn   = softmax(scores, dim=-1)   -> row-wise softmax
  out    = attn @ V                  -> (batch, heads, seq, dim)

Input layout after get_nki_inputs transposition:
  Q_T: (batch, heads, dim, seq)  -- Q transposed on last two dims
  K_T: (batch, heads, dim, seq)  -- K transposed on last two dims
  V:   (batch, heads, seq, dim)  -- V unchanged

Matmul #1: nc_matmul(Q_T_tile, K_T_tile) = Q_T^T @ K_T = Q @ K^T  -> (seq, seq)
  Q_T_tile has par_dim=dim, free=seq; K_T_tile has par_dim=dim, free=seq.

Matmul #2: nc_matmul(attn_T_tile, V_tile) = attn_T^T @ V = attn @ V -> (seq, dim)
  attn_T_tile has par_dim=seq, free=seq; V_tile has par_dim=seq, free=dim.
  We transpose attn (seq, seq) before this matmul.
"""

import numpy as np
import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

SEED = 42


def compute_gold(dims):
    torch.manual_seed(SEED)
    batch, heads, seq, dim = dims["batch"], dims["heads"], dims["seq"], dims["dim"]
    Q = torch.randn(batch, heads, seq, dim)
    K = torch.randn(batch, heads, seq, dim)
    V = torch.randn(batch, heads, seq, dim)
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V)


def get_nki_inputs(dims):
    torch.manual_seed(SEED)
    batch, heads, seq, dim = dims["batch"], dims["heads"], dims["seq"], dims["dim"]
    Q = torch.randn(batch, heads, seq, dim).numpy()
    K = torch.randn(batch, heads, seq, dim).numpy()
    V = torch.randn(batch, heads, seq, dim).numpy()
    # Transpose Q, K on last two dims: (batch, heads, seq, dim) -> (batch, heads, dim, seq)
    Q_T = np.ascontiguousarray(Q.transpose(0, 1, 3, 2))
    K_T = np.ascontiguousarray(K.transpose(0, 1, 3, 2))
    return [Q_T, K_T, V]


@nki.jit
def nki_kernel(Q_T, K_T, V):
    """Scaled dot-product attention.

    Q_T: (batch, heads, dim, seq) -- transposed Q
    K_T: (batch, heads, dim, seq) -- transposed K
    V:   (batch, heads, seq, dim)
    """
    batch, heads, dim, seq = Q_T.shape

    scale = 1.0 / np.sqrt(np.float32(dim))

    out = nl.ndarray((batch, heads, seq, dim), dtype=np.float32, buffer=nl.shared_hbm)

    p_idx_dim = nl.arange(dim)[:, None]
    f_idx_seq = nl.arange(seq)[None, :]
    p_idx_seq = nl.arange(seq)[:, None]
    f_idx_dim = nl.arange(dim)[None, :]

    for b in nl.affine_range(batch):
        for h in nl.affine_range(heads):
            # --- Matmul #1: scores = Q @ K^T, shape (seq, seq) ---
            # nc_matmul(Q_T_tile, K_T_tile) where both are (dim, seq)
            # = Q_T^T @ K_T = Q @ K^T
            q_tile = nl.load(Q_T[b, h, p_idx_dim, f_idx_seq])  # (dim, seq)
            k_tile = nl.load(K_T[b, h, p_idx_dim, f_idx_seq])  # (dim, seq)

            # scores in psum: (seq, seq)
            scores = nisa.nc_matmul(q_tile, k_tile)

            # Scale by 1/sqrt(dim)
            scores_scaled = nisa.tensor_scalar(
                data=scores, op0=np.multiply, operand0=np.float32(scale),
                dtype=np.float32
            )

            # --- Softmax over last dim (each row of seq x seq) ---
            # Phase 1: row-wise max
            row_max = nisa.tensor_reduce(
                data=scores_scaled, op=np.max, axis=(1,), dtype=np.float32
            )  # (seq, 1)

            # Phase 2: exp(scores - max) and row sum
            neg_max = nisa.tensor_scalar(
                data=row_max, op0=np.multiply, operand0=np.float32(-1.0),
                dtype=np.float32
            )
            exp_scores = nisa.activation(
                op=nl.exp, data=scores_scaled, bias=neg_max, scale=1.0,
                dtype=np.float32
            )  # (seq, seq)
            row_sum = nisa.tensor_reduce(
                data=exp_scores, op=np.add, axis=(1,), dtype=np.float32
            )  # (seq, 1)

            # Phase 3: normalize
            inv_sum = nisa.reciprocal(data=row_sum, dtype=np.float32)
            attn = nisa.tensor_scalar(
                data=exp_scores, op0=np.multiply, operand0=inv_sum,
                dtype=np.float32
            )  # (seq, seq)

            # --- Matmul #2: out = attn @ V, shape (seq, dim) ---
            # nc_matmul(attn_T, V) where attn_T is (seq, seq) and V is (seq, dim)
            # = attn_T^T @ V = attn @ V
            # Need to transpose attn: (seq_par, seq_free) -> (seq_par, seq_free)
            # attn is already (seq, seq) with par_dim=seq, free=seq
            # We need attn^T as lhs: par_dim=seq, free=seq -- same shape but transposed
            # Load V from HBM
            v_tile = nl.load(V[b, h, p_idx_seq, f_idx_dim])  # (seq, dim)

            # attn has shape (seq_par, seq_free). We need attn^T as (seq_par, seq_free).
            # For a square matrix, transpose swaps par and free dims.
            # Use nc_matmul with is_transpose=True on the stationary (lhs) to implicitly transpose.
            # Actually, let's just transpose attn explicitly.
            # nc_matmul(lhs, rhs) = lhs^T @ rhs
            # We want attn @ V. lhs^T = attn => lhs = attn^T.
            # lhs must be (par_dim, free) = (seq, seq). We need the transposed version.
            # Since attn is (seq_p, seq_f), attn^T swaps the partition/free axes.
            # We can use nisa.nc_transpose to transpose.

            # For the second matmul, V needs par_dim first.
            # V is (seq, dim). For nc_matmul, rhs needs (par_dim, free).
            # We want result (seq, dim), so rhs par_dim = seq, free = dim.
            # V is already (seq, dim) -- good.

            # For lhs: we need attn^T with (par_dim=seq, free=seq).
            # attn is currently in sbuf as (par=seq, free=seq).
            # Transpose it: swap par and free.
            attn_T = nisa.nc_transpose(attn)  # (seq, seq) transposed

            output_tile = nisa.nc_matmul(attn_T, v_tile)  # (seq, dim)

            nl.store(
                out[b, h, p_idx_seq, f_idx_dim],
                value=output_tile,
            )

    return out
