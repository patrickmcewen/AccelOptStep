"""NKI baseline for mamba2: Mamba2 structured state space model.

Implements the full Mamba2 SSD forward pass in numpy via run_nki(dims).
The computation involves chunked sequence processing with segment sums,
exponential decay, and multiple einsum contractions.
"""

import numpy as np
import torch

SEED = 42


def run_nki(dims):
    """Full Mamba2 forward pass in numpy, matching the PyTorch reference."""
    batch_size = dims["batch_size"]
    seq_length = dims["seq_length"]
    n_heads = dims["n_heads"]
    d_head = dims["d_head"]
    d_state = dims["d_state"]
    block_len = dims["block_len"]

    n_chunks = seq_length // block_len

    # Extract model parameters (same seed as problem)
    torch.manual_seed(SEED)
    from StepBench.problems.mamba2 import Model, get_init_inputs
    model = Model(*get_init_inputs(dims))
    A = model.A.detach().numpy()    # (batch_size, seq_length, n_heads)
    B = model.B.detach().numpy()    # (batch_size, seq_length, n_heads, d_state)
    C = model.C.detach().numpy()    # (batch_size, seq_length, n_heads, d_state)

    # Generate input with same seed
    torch.manual_seed(SEED)
    X = torch.randn(batch_size, seq_length, n_heads, d_head).numpy()

    # Rearrange into blocks: "b (c l) ... -> b c l ..."
    X_blocks = X.reshape(batch_size, n_chunks, block_len, n_heads, d_head)
    A_blocks_raw = A.reshape(batch_size, n_chunks, block_len, n_heads)
    B_blocks = B.reshape(batch_size, n_chunks, block_len, n_heads, d_state)
    C_blocks = C.reshape(batch_size, n_chunks, block_len, n_heads, d_state)

    # A_blocks: "b c l h -> b h c l"
    A_blocks = np.transpose(A_blocks_raw, (0, 3, 1, 2))  # (b, h, c, l)
    A_cumsum = np.cumsum(A_blocks, axis=-1)               # (b, h, c, l)

    # --- 1. Diagonal block outputs ---
    # segsum on A_blocks: shape (b, h, c, l) -> (b, h, c, l, l)
    L = np.exp(_segsum(A_blocks))
    # Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp")
    # Note: s and l are both block_len indices
    Y_diag = np.einsum("bclhn,bcshn,bhcls,bcshp->bclhp",
                        C_blocks, B_blocks, L, X_blocks)

    # --- 2. Intra-chunk states ---
    # decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)  shape (b, h, c, l)
    decay_states = np.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    # states = einsum("bclhn,bhcl,bclhp->bchpn")
    states = np.einsum("bclhn,bhcl,bclhp->bchpn",
                        B_blocks, decay_states, X_blocks)

    # --- 3. Inter-chunk recurrence ---
    # initial_states: zeros like states[:, :1]
    initial_states = np.zeros_like(states[:, :1])  # (b, 1, h, d_head, d_state)
    states_cat = np.concatenate([initial_states, states], axis=1)  # (b, c+1, h, d_head, d_state)

    # F.pad(A_cumsum[:,:,:,-1], (1,0)) pads last dim with 1 zero on the left
    A_last = A_cumsum[:, :, :, -1]  # (b, h, c)
    A_last_padded = np.pad(A_last, ((0, 0), (0, 0), (1, 0)))  # (b, h, c+1)

    decay_chunk = np.exp(_segsum(A_last_padded))  # (b, h, c+1, c+1)
    # new_states = einsum("bhzc,bchpn->bzhpn")
    new_states = np.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_cat)
    states_final = new_states[:, :-1]  # (b, c, h, d_head, d_state)

    # --- 4. State-to-output ---
    state_decay_out = np.exp(A_cumsum)  # (b, h, c, l)
    # Y_off = einsum("bclhn,bchpn,bhcl->bclhp")
    Y_off = np.einsum("bclhn,bchpn,bhcl->bclhp",
                       C_blocks, states_final, state_decay_out)

    # Combine and reshape: "b c l h p -> b (c l) h p"
    Y = (Y_diag + Y_off).reshape(batch_size, seq_length, n_heads, d_head)
    return Y


def _segsum(x):
    """Segment sum: cumsum differences with lower-triangular mask.

    Input x: (..., T)
    Output: (..., T, T) where out[..., i, j] = cumsum[..., i] - cumsum[..., j]
    for i >= j, else -inf.
    """
    T = x.shape[-1]
    x_cumsum = np.cumsum(x, axis=-1)
    # x_cumsum[..., :, None] - x_cumsum[..., None, :] -> (..., T, T)
    x_segsum = x_cumsum[..., :, np.newaxis] - x_cumsum[..., np.newaxis, :]
    mask = np.tril(np.ones((T, T), dtype=bool))
    x_segsum = np.where(mask, x_segsum, -np.inf)
    return x_segsum
