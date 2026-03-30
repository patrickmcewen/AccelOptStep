"""Cycle-accurate correctness verification for NKIBench STeP baselines.

Usage:
    pytest tests/test_nkibench.py -v
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.environ.get("STEP_ARTIFACT_SRC", "/root/step_artifact/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim import simulate, HBMConfig, SimConfig
from utils.gold_checking import reconstruct_numpy
from NKIBench.loader import load_baseline, get_dims


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
    baseline = load_baseline(bench_name)
    assert baseline is not None, f"{bench_name} has no baseline"

    gold = baseline.compute_gold(dims)
    num_stages = getattr(baseline, "STAGES", 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_dir = os.getcwd()
        os.chdir(tmpdir)
        try:
            prev_output = None
            output_op = None
            total_cycles = 0

            for stage in range(1, num_stages + 1):
                pb_path = os.path.join(tmpdir, f"graph_s{stage}.pb")

                if num_stages == 1:
                    graph, output_op = baseline.build_graph(dims)
                else:
                    graph, output_op = baseline.build_graph(
                        dims, stage=stage, prev_output=prev_output
                    )

                cycles, duration_ms, duration_s = simulate(
                    graph,
                    logging=False,
                    hbm_config=HBM,
                    sim_config=SIM,
                    protobuf_file=pb_path,
                    db_name=None,
                )
                total_cycles += cycles

                if stage < num_stages:
                    prev_output = reconstruct_numpy(
                        output_op.store_file_name, delete_npy=False
                    )

            store_name = output_op.store_file_name
            sim_output = reconstruct_numpy(store_name, delete_npy=False)
            sim_tensor = torch.from_numpy(sim_output).float()
        finally:
            os.chdir(orig_dir)

    # Flatten both and compare (baselines may produce different shapes)
    sim_flat = sim_tensor.reshape(-1)
    gold_flat = gold.float().reshape(-1)
    assert sim_flat.numel() == gold_flat.numel(), (
        f"Element count mismatch: sim={sim_flat.numel()} gold={gold_flat.numel()}"
    )

    max_diff = (sim_flat - gold_flat).abs().max().item()
    passed = torch.allclose(sim_flat, gold_flat, rtol=RTOL, atol=ATOL)

    return {
        "passed": passed,
        "cycles": total_cycles,
        "max_diff": max_diff,
        "sim_shape": tuple(sim_tensor.shape),
        "gold_shape": tuple(gold.shape),
    }


def _run_and_assert(bench_name, preset="small"):
    print(f"\n=== {bench_name} ({preset}) ===")
    result = verify_baseline(bench_name, preset)
    print(f"  Cycles: {result['cycles']}")
    print(f"  Sim shape: {result['sim_shape']}, Gold shape: {result['gold_shape']}")
    print(f"  Max absolute diff: {result['max_diff']}")
    assert result["passed"], (
        f"FAILED: max_diff={result['max_diff']}, "
        f"shapes sim={result['sim_shape']} gold={result['gold_shape']}"
    )
    print("  PASSED")


def test_matmul():
    _run_and_assert("matmul")


def test_transpose_matmul():
    _run_and_assert("transpose_matmul")


def test_bmm():
    _run_and_assert("bmm")


def test_silu():
    _run_and_assert("silu")


def test_lora():
    _run_and_assert("lora")


def test_bmm_softmax():
    _run_and_assert("bmm_softmax")


def test_rmsnorm_matmul():
    _run_and_assert("rmsnorm_matmul")


def test_add_rmsnorm_matmul():
    _run_and_assert("add_rmsnorm_matmul")


def test_matmul_add_rmsnorm():
    _run_and_assert("matmul_add_rmsnorm")


def test_swiglu():
    _run_and_assert("swiglu")


def test_rope():
    _run_and_assert("rope_single_freq_apply")


def test_adamw():
    _run_and_assert("adamw")


def test_gqa_full():
    _run_and_assert("gqa_full")
