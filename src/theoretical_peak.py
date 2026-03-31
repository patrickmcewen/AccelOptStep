"""Theoretical peak latency calculation for NKI and StepBench benchmarks on Trainium.

Computes the minimum possible latency for a given problem based on hardware
limits: memory bandwidth, vector FLOPS, and matmul FLOPS. Returns which
resource is the bottleneck and the corresponding peak latency.

Ported from AccelOpt/experiments/full_complete_local/calculate_percentage_of_peak.py
"""

PEAK_BW = 410 * 1024 * 1024 * 1024            # 410 GiB/s
PEAK_VEC_FLOPS = 2 * 128 * 1.12 * 1e9         # 286.8 GFLOPS (Vec + Scalar)
PEAK_MATMUL_FLOPS = 23.75 * 1e12              # 23.75 TFLOPS


def calc_theoretical_peak(problem_name: str, config: dict) -> dict | None:
    """Compute theoretical peak latency for a problem on Trainium.

    Args:
        problem_name: Benchmark problem name (e.g. "matmul", "silu").
        config: Dimension dict (e.g. {"M": 4096, "N": 12288, "K": 5120}).

    Returns:
        Dict with "theoretical_peak_latency" (ms) and "bound_key" ("vec"/"mm"/"memory"),
        or None if the problem is not recognized.
    """
    vec_flops = 0
    mm_flops = 0
    total_bytes = 0

    if problem_name == "mamba":
        M, C, S = config["M"], config["C"], config["S"]
        vec_flops = M * C * S * (2 + 2) + M * (C * S * 2) + M * C * S * 2
        total_bytes = (C * M * 2 + C * S + S * M * 2 + C * M) * 4
    elif problem_name == "silu":
        M, N = config["M"], config["N"]
        vec_flops = M * N * 3
        total_bytes = (M * N + N * M) * 4
    elif problem_name == "add_rmsnorm_matmul":
        M, N, K = config["M"], config["N"], config["K"]
        vec_flops = M * K * 6
        mm_flops = M * N * K * 2
        total_bytes = (M * K * 2 + K * N + K + M * N) * 4
    elif problem_name == "matmul_add_rmsnorm":
        M, N, K = config["M"], config["N"], config["K"]
        vec_flops = M * N * 6
        mm_flops = M * N * K * 2
        total_bytes = (M * K + N * K + M * N + N + M * N) * 4
    elif problem_name == "rmsnorm_matmul":
        M, N, K = config["M"], config["N"], config["K"]
        vec_flops = M * K * 4
        mm_flops = M * N * K * 2
        total_bytes = (M * K + K * N + M * N) * 4
    elif problem_name == "swiglu":
        M, N, K = config["M"], config["N"], config["K"]
        vec_flops = M * N * 5
        mm_flops = M * K * N * 2 * 3
        total_bytes = (M * K * 2 + K * N * 3) * 4
    elif problem_name == "matmul":
        M, N, K = config["M"], config["N"], config["K"]
        mm_flops = M * N * K * 2
        total_bytes = (M * N + N * K + M * K) * 4
    elif problem_name == "gqa_full":
        B, N, QH, KH, D = config["B"], config["N"], config["QH"], config["KH"], config["D"]
        mm_flops = B * QH * N * D * N * 2 * 2
        vec_flops = B * QH * N * N * 5
        total_bytes = (B * QH * N * D * 2 + B * KH * N * D * 2) * 4
    elif problem_name == "rope_single_freq_apply":
        B, H, N, D = config["B"], config["H"], config["N"], config["D"]
        vec_flops = B * H * N * D * 3
        total_bytes = (D * B * H * N) * 3 * 4
    elif problem_name == "bmm":
        B, M, N, K = config["B"], config["M"], config["N"], config["K"]
        mm_flops = B * M * N * K * 2
        total_bytes = (B * M * N + B * N * K + B * M * K) * 4
    elif problem_name == "bmm_softmax":
        B, M, N, K = config["B"], config["M"], config["N"], config["K"]
        vec_flops = B * M * N * 4 + B * M * N
        mm_flops = B * M * N * K * 2
        total_bytes = (B * M * N + B * N * K + B * M * K) * 4
    elif problem_name == "transpose_matmul":
        M, N, K = config["M"], config["N"], config["K"]
        mm_flops = M * N * K * 2
        total_bytes = (M * N + N * K + M * K) * 4
    elif problem_name == "lora":
        M, N, K, R = config["M"], config["N"], config["K"], config["R"]
        vec_flops = M * N
        mm_flops = M * N * K * 2 + M * K * R * 2 + M * N * R * 2
        total_bytes = (M * K + K * N + K * R + R * N) * 4 + M * N * 4
    elif problem_name == "adamw":
        M, N = config["M"], config["N"]
        vec_flops = M * N * (1 + 1 + 2 + 3)
        total_bytes = M * N * 4 * 4 + M * N * 4

    # ---- StepBench benchmarks ------------------------------------------------
    elif problem_name == "gemm_batched":
        batch, M, K, N = config["batch"], config["M"], config["K"], config["N"]
        mm_flops = batch * M * N * K * 2
        total_bytes = (batch * M * K + batch * K * N + batch * M * N) * 4

    elif problem_name == "gemm_3d":
        N, M, K, L = config["N"], config["M"], config["K"], config["L"]
        mm_flops = N * M * L * K * 2
        total_bytes = (N * M * K + K * L + N * M * L) * 4

    elif problem_name == "gemm_scale_residual":
        # y = Linear(x) * scaling_factor + Linear(x)  (same linear, so one matmul)
        B = config["batch_size"]
        I, O = config["in_features"], config["out_features"]
        mm_flops = B * I * O * 2
        vec_flops = B * O * 2          # scale + add
        total_bytes = (B * I + I * O + O + B * O) * 4   # x, weight, bias, output

    elif problem_name == "gemm_swish_scaling":
        # y = swish(x @ W^T) * scaling_factor
        B = config["batch_size"]
        I, O = config["in_features"], config["out_features"]
        mm_flops = B * I * O * 2
        vec_flops = B * O * 4          # sigmoid + mul (swish) + scale
        total_bytes = (B * I + I * O + B * O) * 4       # x, weight (no bias), output

    elif problem_name == "activation":
        B, D = config["batch_size"], config["dim"]
        fn = config.get("fn", "relu")
        elems = B * D
        # flop count varies by activation
        if fn == "relu":
            vec_flops = elems              # compare
        elif fn == "sigmoid":
            vec_flops = elems * 4          # exp, add, div, neg
        elif fn == "gelu":
            vec_flops = elems * 6          # tanh-approx: pow, mul, add, tanh, add, mul
        elif fn == "swish":
            vec_flops = elems * 5          # sigmoid(4) + mul
        elif fn == "softmax":
            vec_flops = elems * 5          # max, sub, exp, sum, div
        else:
            vec_flops = elems * 4          # generic fallback
        total_bytes = (elems + elems) * 4  # read input + write output

    elif problem_name == "layernorm":
        B = config["batch_size"]
        F_ = config["features"]
        D1, D2 = config["dim1"], config["dim2"]
        elems = B * F_ * D1 * D2
        norm_size = F_ * D1 * D2
        # mean, variance, normalize, scale, shift
        vec_flops = elems * 5
        # input + output + gamma + beta
        total_bytes = (elems + elems + norm_size + norm_size) * 4

    elif problem_name == "rmsnorm":
        # x / sqrt(mean(x^2, dim=1) + eps)
        B = config["batch_size"]
        F_ = config["features"]
        D1, D2 = config["dim1"], config["dim2"]
        elems = B * F_ * D1 * D2
        # square, reduce-mean over features, sqrt, divide
        vec_flops = elems * 4
        total_bytes = (elems + elems) * 4  # input + output

    elif problem_name == "sdpa":
        # scaled_dot_product_attention(Q, K, V)
        B, H, S, D = config["batch"], config["heads"], config["seq"], config["dim"]
        # Q@K^T: B*H * S*S*D*2, softmax: B*H*S*S*5, attn@V: B*H * S*D*S*2
        mm_flops = B * H * S * S * D * 2 * 2       # two matmuls
        vec_flops = B * H * S * S * 5               # softmax + scale
        # Q, K, V in + output
        total_bytes = (3 * B * H * S * D + B * H * S * D) * 4

    elif problem_name == "minigpt_block":
        # Full transformer block: LN + Attn(QKV proj, attn, out proj) + LN + FFN(4x expand, GELU, project)
        B = config["batch_size"]
        T = config["seq_len"]
        C = config["n_embd"]
        nh = config["n_head"]
        hs = C // nh
        # Attention:
        #   c_attn: B*T*C -> B*T*3C  (mm: B*T*3C*C*2)
        #   Q@K^T: B*nh*T*T*hs*2,  attn@V: B*nh*T*T*hs*2
        #   c_proj: B*T*C*C*2
        mm_attn = B * T * 3 * C * C * 2 + B * nh * T * T * hs * 2 * 2 + B * T * C * C * 2
        # softmax + scale on T*T attention scores
        vec_attn = B * nh * T * T * 6
        # FFN:
        #   c_fc: B*T*C -> B*T*4C  (mm: B*T*4C*C*2)
        #   GELU: B*T*4C*6
        #   c_proj: B*T*4C -> B*T*C  (mm: B*T*C*4C*2)
        mm_ffn = B * T * 4 * C * C * 2 + B * T * C * 4 * C * 2
        vec_ffn = B * T * 4 * C * 6     # GELU
        # LayerNorm x2: B*T*C*5 each
        vec_ln = 2 * B * T * C * 5
        # Residuals: 2 adds
        vec_res = 2 * B * T * C

        mm_flops = mm_attn + mm_ffn
        vec_flops = vec_attn + vec_ffn + vec_ln + vec_res
        # x input + output (same shape), weights are on-chip constants for peak calc
        # Conservative: just count input/output activations
        total_bytes = (B * T * C + B * T * C) * 4   # in + out
        # Also count weight reads: c_attn(C*3C), c_proj(C*C), c_fc(C*4C), c_proj_ffn(4C*C), ln params
        total_bytes += (C * 3 * C + C * C + C * 4 * C + 4 * C * C) * 4
        total_bytes += 2 * C * 2 * 4  # two layernorm gamma+beta

    elif problem_name == "moe":
        # Top-1 MoE: gate linear + num_experts expert linears (all tokens go through exactly 1 expert)
        NT = config["num_tokens"]
        H = config["hidden_dim"]
        I = config["intermediate_dim"]
        E = config["num_experts"]
        # Gate: NT*H -> NT*E
        mm_gate = NT * H * E * 2
        # Experts: total tokens = NT, each goes to one expert (same total work as one big linear)
        mm_experts = NT * H * I * 2
        mm_flops = mm_gate + mm_experts
        vec_flops = NT * E            # argmax over gate scores
        # gate weight + expert weights + input + output
        total_bytes = (H * E + E * H * I + NT * H + NT * I) * 4

    elif problem_name == "mlp":
        # Linear+ReLU stack: input -> [hidden layers with ReLU] -> output
        B = config["batch_size"]
        sizes = config["layer_sizes"]
        inp = config["input_size"]
        out = config["output_size"]
        prev = inp
        for s in sizes:
            mm_flops += B * prev * s * 2
            vec_flops += B * s           # ReLU
            total_bytes += (prev * s + s) * 4  # weight + bias
            prev = s
        # Final linear (no ReLU)
        mm_flops += B * prev * out * 2
        total_bytes += (prev * out + out) * 4  # weight + bias
        # Input + output activations
        total_bytes += (B * inp + B * out) * 4

    elif problem_name == "mlp_shallow_wide":
        # Same architecture as mlp
        B = config["batch_size"]
        sizes = config["hidden_layer_sizes"]
        inp = config["input_size"]
        out = config["output_size"]
        prev = inp
        for s in sizes:
            mm_flops += B * prev * s * 2
            vec_flops += B * s           # ReLU
            total_bytes += (prev * s + s) * 4
            prev = s
        mm_flops += B * prev * out * 2
        total_bytes += (prev * out + out) * 4
        total_bytes += (B * inp + B * out) * 4

    elif problem_name == "mamba2":
        # Mamba2 SSD: chunked structured state space
        B = config["batch_size"]
        T = config["seq_length"]
        H = config["n_heads"]
        D = config["d_head"]
        S = config["d_state"]
        L = config["block_len"]
        C = T // L  # number of chunks
        # Diagonal block: einsum "bclhn,bcshn,bhcls,bcshp->bclhp"
        #   B_blocks(B,C,L,H,S) * C_blocks * L_mat(B,H,C,L,L) * X_blocks(B,C,L,H,D)
        #   ~ B*C * H * L*L * (S + D) * 2
        mm_diag = B * C * H * L * L * (S + D) * 2
        # States: einsum "bclhn,bhcl,bclhp->bchpn" ~ B*C*H * L * D * S * 2
        mm_states = B * C * H * L * D * S * 2
        # Inter-chunk decay: einsum "bhzc,bchpn->bzhpn" ~ B*H * C*C * D * S * 2
        mm_inter = B * H * C * C * D * S * 2
        # State-to-output: einsum "bclhn,bchpn,bhcl->bclhp" ~ B*C*H * L * D * S * 2
        mm_out = B * C * H * L * D * S * 2
        mm_flops = mm_diag + mm_states + mm_inter + mm_out
        # segsum, cumsum, exp operations
        vec_flops = B * H * C * L * L * 3 + B * H * C * L * 3 + B * C * H * L * D * 2
        # Parameters A(B,T,H), B_param(B,T,H,S), C_param(B,T,H,S), X(B,T,H,D), Y(B,T,H,D)
        total_bytes = (B * T * H + B * T * H * S * 2 + B * T * H * D * 2) * 4

    else:
        return None

    latency = {
        "vec": vec_flops / PEAK_VEC_FLOPS,
        "mm": mm_flops / PEAK_MATMUL_FLOPS,
        "memory": total_bytes / PEAK_BW,
    }
    bound_key = max(latency, key=latency.get)
    return {
        "theoretical_peak_latency": latency[bound_key] * 1e3,  # seconds -> ms
        "bound_key": bound_key,
    }
