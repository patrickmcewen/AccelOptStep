"""Cycle-accurate correctness verification for STeP baselines.

Usage:
    pytest tests/test_correctness_small.py -v
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.environ.get("STEP_ARTIFACT_SRC", "/root/step_artifact/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim import simulate, HBMConfig, SimConfig
from utils.gold_checking import reconstruct_numpy
from StepBench.loader import load_baseline, load_problem, get_dims


HBM = HBMConfig(
    addr_offset=64,
    channel_num=32,
    per_channel_latency=2,
    per_channel_init_interval=2,
    per_channel_outstanding=1,
    per_channel_start_up_time=14,
)

SIM = SimConfig(
    channel_depth=2,
    functional_sim=True,
    mock_bf16=False,
)

RTOL = 1e-3
ATOL = 1e-3


def verify_baseline(bench_name, preset="small"):
    """Run cycle-accurate simulation on a baseline and compare against gold."""
    dims = get_dims(bench_name, preset)
    problem = load_problem(bench_name)
    baseline = load_baseline(bench_name)
    assert baseline is not None, f"{bench_name} has no baseline"

    gold = problem.compute_gold(dims)
    graph, output_op = baseline.build_graph(dims)
    # Some baselines permute dimensions for efficient tiling; read the inverse permute info
    output_permute = getattr(baseline, "OUTPUT_PERMUTE", None)

    with tempfile.TemporaryDirectory() as tmpdir:
        pb_path = os.path.join(tmpdir, "graph.pb")
        orig_dir = os.getcwd()
        os.chdir(tmpdir)

        cycles, duration_ms, duration_s = simulate(
            graph,
            logging=False,
            hbm_config=HBM,
            sim_config=SIM,
            protobuf_file=pb_path,
            db_name=None,
        )

        store_name = output_op.store_file_name
        sim_output = reconstruct_numpy(store_name, delete_npy=False)
        sim_tensor = torch.from_numpy(sim_output).float()

        os.chdir(orig_dir)

    # Sim output is often flat 2D; reshape to match gold if element counts agree.
    # If the baseline specifies OUTPUT_PERMUTE, reshape to intermediate shape first,
    # then permute to match gold layout.
    if sim_tensor.shape != gold.shape and sim_tensor.numel() == gold.numel():
        if output_permute is not None:
            # Compute inverse permute to get intermediate shape
            inv_perm = [0] * len(output_permute)
            for i, p in enumerate(output_permute):
                inv_perm[p] = i
            intermediate_shape = tuple(gold.shape[inv_perm[i]] for i in range(len(gold.shape)))
            sim_tensor = sim_tensor.reshape(intermediate_shape).permute(output_permute)
        else:
            sim_tensor = sim_tensor.reshape(gold.shape)

    max_diff = (sim_tensor.float() - gold.float()).abs().max().item()
    passed = sim_tensor.shape == gold.shape and torch.allclose(
        sim_tensor.float(), gold.float(), rtol=RTOL, atol=ATOL
    )

    return {
        "passed": passed,
        "cycles": cycles,
        "duration_ms": duration_ms,
        "max_diff": max_diff,
        "sim_shape": tuple(sim_tensor.shape),
        "gold_shape": tuple(gold.shape),
    }


def _run_and_assert(bench_name, preset="small"):
    print(f"\n=== {bench_name} ({preset}) ===")
    result = verify_baseline(bench_name, preset)
    print(f"  Cycles: {result['cycles']}, Duration: {result['duration_ms']}ms")
    print(f"  Sim shape: {result['sim_shape']}, Gold shape: {result['gold_shape']}")
    print(f"  Max absolute diff: {result['max_diff']}")
    assert result["passed"], (
        f"FAILED: max_diff={result['max_diff']}, "
        f"shapes sim={result['sim_shape']} gold={result['gold_shape']}"
    )
    print("  PASSED")


def test_gemm():
    _run_and_assert("gemm", "small")


def test_gemm_swish_scaling():
    _run_and_assert("gemm_swish_scaling", "small")


def test_sdpa():
    _run_and_assert("sdpa", "small")


def test_gemm_batched():
    _run_and_assert("gemm_batched", "small")


def test_gemm_3d():
    _run_and_assert("gemm_3d", "small")


def test_activation_swish():
    _run_and_assert("activation", "swish_small")


def test_activation_softmax():
    _run_and_assert("activation", "softmax_small")


def test_gemm_scale_residual():
    _run_and_assert("gemm_scale_residual", "small")


def test_rmsnorm():
    _run_and_assert("rmsnorm", "small")


def test_layernorm():
    _run_and_assert("layernorm", "small")
