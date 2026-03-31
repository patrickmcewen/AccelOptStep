"""Theoretical peak latency calculation for NKI benchmarks on Trainium.

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
