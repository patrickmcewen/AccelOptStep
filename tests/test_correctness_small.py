"""Cycle-accurate correctness verification for STeP baselines.

Usage:
    # Verify a single baseline (full size — may be slow):
    python tests/test_correctness_small.py StepBench.baselines.gemm_square

    # Verify with smaller tensors for fast iteration:
    python tests/test_correctness_small.py StepBench.baselines.gemm_square --small

    # Verify all starter baselines:
    python tests/test_correctness_small.py --all --small

    # Use as pytest (runs all three starter baselines at reduced size):
    pytest tests/test_correctness_small.py -v

Each baseline module must export:
    build_graph() -> (networkx.MultiDiGraph, OffChipStore)
    compute_gold() -> torch.Tensor

And must use module-level dimension constants (M, K, N, etc.) that both
build_graph() and compute_gold() read, so the test can patch them for
fast verification at reduced sizes.
"""
import argparse
import importlib
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.environ.get(
    "STEP_ARTIFACT_SRC", "/root/step_artifact/src"
))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim import simulate, HBMConfig, SimConfig
from utils.gold_checking import reconstruct_numpy


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

# Per-baseline dimension overrides for --small mode.
# Keys are module paths; values are dicts of {attr_name: small_value}.
# Only GEMM-family baselines support dimension shrinking cleanly.
# SDPA has coupled dimensions that don't shrink safely, so it runs full-size.
SMALL_OVERRIDES = {
    "StepBench.baselines.gemm_square": {"M": 256, "K": 256, "N": 256},
    "StepBench.baselines.gemm_silu": {"M": 256, "K": 256, "N": 256},
    "StepBench.baselines.sdpa": {"batch": 1, "heads": 1, "seq": 64, "dim": 128},
}


def _shrink_module(mod, module_path: str):
    """Patch module-level dimension constants to small values for fast sim."""
    overrides = SMALL_OVERRIDES.get(module_path, {})
    for name, small_val in overrides.items():
        assert hasattr(mod, name), f"{module_path} has no attribute {name}"
        setattr(mod, name, small_val)


def verify_baseline(module_path: str, small: bool = False) -> dict:
    """Run cycle-accurate simulation on a baseline and compare against gold.

    Args:
        module_path: dotted Python module path, e.g. "StepBench.baselines.gemm_square"
        small: if True, patch module dimensions to small values for fast sim

    Returns:
        dict with keys: passed, cycles, duration_ms, max_diff, sim_shape, gold_shape
    """
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "build_graph"), f"{module_path} must export build_graph()"
    assert hasattr(mod, "compute_gold"), f"{module_path} must export compute_gold()"

    if small:
        _shrink_module(mod, module_path)

    gold = mod.compute_gold()
    graph, output_op = mod.build_graph()

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


# ---- pytest entry points (always use small mode) ----

STARTER_BASELINES = [
    "StepBench.baselines.gemm_square",
    "StepBench.baselines.gemm_silu",
    "StepBench.baselines.sdpa",
]


def _run_and_assert(module_path: str, small: bool = True):
    print(f"\n=== {module_path}{' (small)' if small else ''} ===")
    result = verify_baseline(module_path, small=small)
    print(f"  Cycles: {result['cycles']}, Duration: {result['duration_ms']}ms")
    print(f"  Sim shape: {result['sim_shape']}, Gold shape: {result['gold_shape']}")
    print(f"  Max absolute diff: {result['max_diff']}")
    assert result["passed"], (
        f"FAILED: max_diff={result['max_diff']}, "
        f"shapes sim={result['sim_shape']} gold={result['gold_shape']}"
    )
    print("  PASSED")


def test_gemm_square():
    _run_and_assert("StepBench.baselines.gemm_square", small=True)


def test_gemm_silu():
    _run_and_assert("StepBench.baselines.gemm_silu", small=True)


def test_sdpa():
    _run_and_assert("StepBench.baselines.sdpa", small=True)


# ---- CLI entry point ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "module", nargs="?",
        help="Dotted module path, e.g. StepBench.baselines.gemm_square",
    )
    parser.add_argument("--all", action="store_true", help="Verify all starter baselines")
    parser.add_argument(
        "--small", action="store_true",
        help="Shrink dimensions for fast verification (seconds instead of hours)",
    )
    args = parser.parse_args()

    assert args.module or args.all, "Provide a module path or --all"

    targets = STARTER_BASELINES if args.all else [args.module]

    for target in targets:
        _run_and_assert(target, small=args.small)

    print(f"\n{'='*40}")
    print(f"All {len(targets)} baselines verified successfully.")
