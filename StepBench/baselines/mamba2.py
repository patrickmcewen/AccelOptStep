"""Mamba2 structured state space model baseline in STeP IR.

The Mamba2 SSD computation involves chunked processing with:
  - Diagonal block outputs (Y_diag): intra-chunk attention-like computation
  - Off-diagonal outputs (Y_off): inter-chunk state recurrence

Key decomposition:
  Y_diag per (b,c,h) = (C @ B^T * L) @ X     [two matmuls + elementwise]
  Y_off  per (b,c,h) = (C @ states^T) * decay [one matmul + elementwise]
  Y = Y_diag + Y_off

To produce (L, H*P) output tiles with both heads concatenated,
we use block-diagonal input matrices so a single matmul computes
all heads simultaneously:
  A_combined = [CB_L_h0 | CB_L_h1 | ...]   shape (L, H*L)
  B_combined = blkdiag(X_h0, X_h1, ...)     shape (H*L, H*P)
  Y_diag = A_combined @ B_combined           shape (L, H*P)

Inter-chunk recurrence, segsum, cumsum, and exponentials are
precomputed on the host (deterministic from SEED).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx import MultiDiGraph

from step_py.functions import map_fn
from step_py.ops import OffChipLoad, BinaryMap, OffChipStore
from rewrite.broadcast import infer_broadcast

SEED = 42


def _segsum(x):
    """Segment sum: cumsum outer-difference with lower-triangular mask."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def _precompute(dims):
    """Replicate mamba2 forward pass on CPU, return intermediate values."""
    B = dims["batch_size"]
    S = dims["seq_length"]
    H = dims["n_heads"]
    P = dims["d_head"]
    N = dims["d_state"]
    L = dims["block_len"]
    C = S // L  # number of chunks

    # Replicate model parameters (same RNG as compute_gold)
    torch.manual_seed(SEED)
    A = torch.randn(B, S, H)
    B_param = torch.randn(B, S, H, N)
    C_param = torch.randn(B, S, H, N)

    # Replicate input
    torch.manual_seed(SEED)
    X = torch.randn(B, S, H, P)

    # Chunk: (B, C, L, ...)
    X_blocks = X.reshape(B, C, L, H, P)
    A_raw = A.reshape(B, C, L, H)
    B_blocks = B_param.reshape(B, C, L, H, N)
    C_blocks = C_param.reshape(B, C, L, H, N)

    # A_blocks: (B, H, C, L)
    A_blocks = A_raw.permute(0, 3, 1, 2).contiguous()
    A_cumsum = torch.cumsum(A_blocks, dim=-1)

    # L matrix: exp(segsum(A_blocks)) → (B, H, C, L, L)
    L_mat = torch.exp(_segsum(A_blocks))

    # ========== Y_diag: decomposed einsum ==========
    # Per (b,c,h): CB = C_h @ B_h^T : (L,N)@(N,L) = (L,L)
    #              CB_L = CB * L_h   : (L,L) elementwise
    #              Y_diag_h = CB_L @ X_h : (L,L)@(L,P) = (L,P)
    # Combined: A_diag = [CB_L_h0 | CB_L_h1 | ...] : (L, H*L)
    #           B_diag = blkdiag(X_h0, ...) : (H*L, H*P)

    A_diag_list = []  # (B*C,) of (L, H*L)
    B_diag_list = []  # (B*C,) of (H*L, H*P)

    for b in range(B):
        for c in range(C):
            cb_l_parts = []
            x_parts = []
            for h in range(H):
                C_h = C_blocks[b, c, :, h, :]       # (L, N)
                B_h = B_blocks[b, c, :, h, :]       # (L, N)
                L_h = L_mat[b, h, c, :, :]           # (L, L)
                X_h = X_blocks[b, c, :, h, :]       # (L, P)

                CB_h = C_h @ B_h.T                   # (L, L)
                CB_L_h = CB_h * L_h                  # (L, L)
                cb_l_parts.append(CB_L_h)
                x_parts.append(X_h)

            # A_combined: (L, H*L) — horizontal concat of CB_L per head
            A_diag_list.append(torch.cat(cb_l_parts, dim=1))
            # B_combined: block_diag(X_h0, X_h1, ...) — (H*L, H*P)
            B_diag = torch.zeros(H * L, H * P)
            for h in range(H):
                B_diag[h * L:(h + 1) * L, h * P:(h + 1) * P] = x_parts[h]
            B_diag_list.append(B_diag)

    # Stack into underlying tensors: (B*C * tile_rows, tile_cols)
    A_diag_all = torch.stack(A_diag_list, dim=0).reshape(B * C * L, H * L).contiguous()
    B_diag_all = torch.stack(B_diag_list, dim=0).reshape(B * C * H * L, H * P).contiguous()

    # ========== Y_off: inter-chunk state computation ==========
    # decay_states: exp(A_cumsum[:,:,:,-1:] - A_cumsum) → (B, H, C, L)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)

    # states = einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)
    # Per (b,c,h): states_h = (B_h * decay_h[:,None])^T @ X_h : (N,L)@(L,P) = (N,P)
    # states[b,c,h,p,n] = sum_l B[l,n]*decay[l]*X[l,p]
    # = (B_decay^T @ X)[n,p] → transposed to (P, N)
    states = torch.zeros(B, C, H, P, N)
    for b in range(B):
        for c in range(C):
            for h in range(H):
                B_h = B_blocks[b, c, :, h, :]
                decay_h = decay_states[b, h, c, :]
                X_h = X_blocks[b, c, :, h, :]
                B_decay = B_h * decay_h[:, None]
                # (B_decay^T @ X) is (N, P), states[b,c,h] is (P, N)
                states[b, c, h] = (B_decay.T @ X_h).T

    # Inter-chunk recurrence
    initial_states = torch.zeros(B, 1, H, P, N)
    states_with_init = torch.cat([initial_states, states], dim=1)  # (B, C+1, H, P, N)

    # decay_chunk = exp(segsum(pad(A_cumsum[:,:,:,-1], (1,0))))
    padded = F.pad(A_cumsum[:, :, :, -1], (1, 0))  # (B, H, C+1)
    decay_chunk = torch.exp(_segsum(padded))  # (B, H, C+1, C+1)

    # new_states = einsum("bhzc,bchpn->bzhpn", decay_chunk, states_with_init)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_with_init)

    states_final = new_states[:, :-1]  # (B, C, H, P, N)

    # state_decay_out = exp(A_cumsum) → (B, H, C, L)
    state_decay_out = torch.exp(A_cumsum)

    # Y_off per (b,c,h) = (C_h @ states_h^T) * decay_h
    # C_h: (L, N), states_h: (P, N) → states_h^T: (N, P)
    # result: (L, P) * decay_h[:,None]

    # Combined: C_combined = [C_h0 | C_h1 | ...] : (L, H*N)
    #           states_combined = blkdiag(st_h0^T, ...) : (H*N, H*P)
    #           Y_off_raw = C_combined @ states_combined : (L, H*P)
    #           decay_combined: (L, H*P) — broadcast decay per head across P cols

    C_off_list = []
    S_off_list = []
    decay_off_list = []

    for b in range(B):
        for c in range(C):
            c_parts = []
            s_blk = torch.zeros(H * N, H * P)
            decay_row = torch.zeros(L, H * P)
            for h in range(H):
                C_h = C_blocks[b, c, :, h, :]           # (L, N)
                st_h = states_final[b, c, h]             # (P, N)
                dec_h = state_decay_out[b, h, c, :]      # (L,)

                c_parts.append(C_h)
                s_blk[h * N:(h + 1) * N, h * P:(h + 1) * P] = st_h.T  # (N, P)
                decay_row[:, h * P:(h + 1) * P] = dec_h[:, None].expand(L, P)

            C_off_list.append(torch.cat(c_parts, dim=1))      # (L, H*N)
            S_off_list.append(s_blk)                           # (H*N, H*P)
            decay_off_list.append(decay_row)                   # (L, H*P)

    C_off_all = torch.stack(C_off_list).reshape(B * C * L, H * N).contiguous()
    S_off_all = torch.stack(S_off_list).reshape(B * C * H * N, H * P).contiguous()
    decay_off_all = torch.stack(decay_off_list).reshape(B * C * L, H * P).contiguous()

    return {
        "B_total": B, "C_total": C, "H": H, "L": L, "P": P, "N": N,
        "A_diag": A_diag_all,       # (B*C*L, H*L)
        "B_diag": B_diag_all,       # (B*C*H*L, H*P)
        "C_off": C_off_all,         # (B*C*L, H*N)
        "S_off": S_off_all,         # (B*C*H*N, H*P)
        "decay_off": decay_off_all, # (B*C*L, H*P)
    }


def build_graph(dims):
    pre = _precompute(dims)
    B = pre["B_total"]
    C = pre["C_total"]
    H = pre["H"]
    L = pre["L"]
    P = pre["P"]
    N = pre["N"]

    num_chunks = B * C  # stream length
    par_dispatch = 4
    bw_matmul = 512
    bw_mul = 128
    bw_add = 128

    graph = MultiDiGraph()

    # ===== Y_diag: A_combined @ B_combined =====
    # A_combined: per chunk (L, H*L), stream (num_chunks, 1)
    a_diag_load = OffChipLoad(
        underlying=pre["A_diag"],  # (num_chunks*L, H*L)
        stride=(1, 0),
        out_shape_tiled=(num_chunks, 1),
        tile_row=L, tile_col=H * L,
        par_dispatch=par_dispatch,
    )

    # B_combined: per chunk (H*L, H*P), stream (num_chunks, 1)
    b_diag_load = OffChipLoad(
        underlying=pre["B_diag"],  # (num_chunks*H*L, H*P)
        stride=(1, 0),
        out_shape_tiled=(num_chunks, 1),
        tile_row=H * L, tile_col=H * P,
        par_dispatch=par_dispatch,
    )

    # Y_diag = A @ B : (L, H*L) @ (H*L, H*P) = (L, H*P)
    y_diag = BinaryMap(
        graph=graph, in1=a_diag_load, in2=b_diag_load,
        fn=map_fn.Matmul(weight_transposed=False),
        write_back_mu=False, compute_bw=bw_matmul,
    )

    # ===== Y_off: C_combined @ S_combined * decay =====
    c_off_load = OffChipLoad(
        underlying=pre["C_off"],  # (num_chunks*L, H*N)
        stride=(1, 0),
        out_shape_tiled=(num_chunks, 1),
        tile_row=L, tile_col=H * N,
        par_dispatch=par_dispatch,
    )

    s_off_load = OffChipLoad(
        underlying=pre["S_off"],  # (num_chunks*H*N, H*P)
        stride=(1, 0),
        out_shape_tiled=(num_chunks, 1),
        tile_row=H * N, tile_col=H * P,
        par_dispatch=par_dispatch,
    )

    # C @ S^T gives wrong shape; C @ S directly: (L, H*N) @ (H*N, H*P) = (L, H*P)
    cs = BinaryMap(
        graph=graph, in1=c_off_load, in2=s_off_load,
        fn=map_fn.Matmul(weight_transposed=False),
        write_back_mu=False, compute_bw=bw_matmul,
    )

    # Elementwise multiply by decay
    decay_load = OffChipLoad(
        underlying=pre["decay_off"],  # (num_chunks*L, H*P)
        stride=(1, 0),
        out_shape_tiled=(num_chunks, 1),
        tile_row=L, tile_col=H * P,
        par_dispatch=par_dispatch,
    )

    y_off = BinaryMap(
        graph=graph, in1=cs, in2=decay_load,
        fn=map_fn.Mul(), write_back_mu=False, compute_bw=bw_mul,
    )

    # ===== Y = Y_diag + Y_off =====
    y_total = BinaryMap(
        graph=graph, in1=y_diag, in2=y_off,
        fn=map_fn.Add(), write_back_mu=True, compute_bw=bw_add,
    )

    output_op = OffChipStore(
        graph=graph, input=y_total,
        par_dispatch=par_dispatch, store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
