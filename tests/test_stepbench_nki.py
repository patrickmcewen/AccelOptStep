"""Correctness verification for StepBench NKI baselines.

Compares NKI kernel output (via nki.simulate_kernel) against PyTorch compute_gold().

Usage:
    pytest tests/test_stepbench_nki.py -v
    pytest tests/test_stepbench_nki.py::test_gemm -v
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from StepBench.loader import load_config, load_problem, load_nki_baseline, get_dims

import neuronxcc.nki as nki

RTOL = 1e-3
ATOL = 1e-3


def verify_nki_baseline(bench_name, preset="small"):
    """Run NKI kernel in simulation and compare against PyTorch gold."""
    dims = get_dims(bench_name, preset)
    problem = load_problem(bench_name)
    nki_mod = load_nki_baseline(bench_name)
    assert nki_mod is not None, f"{bench_name} has no NKI baseline"

    # PyTorch gold
    gold = problem.compute_gold(dims)
    if isinstance(gold, torch.Tensor):
        gold_np = gold.detach().float().numpy()
    else:
        gold_np = np.array(gold, dtype=np.float32)

    # NKI kernel execution via simulation
    if hasattr(nki_mod, 'run_nki'):
        # Module provides its own execution (e.g. chaining multiple sub-kernels)
        nki_output = nki_mod.run_nki(dims)
    else:
        nki_inputs = nki_mod.get_nki_inputs(dims)
        if hasattr(nki_mod, 'get_nki_kernel'):
            kernel_fn = nki_mod.get_nki_kernel(dims)
        else:
            kernel_fn = nki_mod.nki_kernel
        nki_output = nki.simulate_kernel(kernel_fn, *nki_inputs)

    if isinstance(nki_output, tuple):
        nki_output = nki_output[0]
    nki_np = np.array(nki_output, dtype=np.float32)

    # Reshape to match if needed
    nki_flat = nki_np.reshape(-1)
    gold_flat = gold_np.reshape(-1)
    assert nki_flat.size == gold_flat.size, (
        f"Element count mismatch: nki={nki_flat.size} gold={gold_flat.size} "
        f"(nki shape={nki_np.shape}, gold shape={gold_np.shape})"
    )

    max_diff = np.max(np.abs(nki_flat - gold_flat))
    passed = np.allclose(nki_flat, gold_flat, rtol=RTOL, atol=ATOL)

    return {
        "passed": passed,
        "max_diff": float(max_diff),
        "nki_shape": nki_np.shape,
        "gold_shape": gold_np.shape,
    }


def _run_and_assert(bench_name, preset="small"):
    print(f"\n=== {bench_name} ({preset}) ===")
    result = verify_nki_baseline(bench_name, preset)
    print(f"  NKI shape: {result['nki_shape']}, Gold shape: {result['gold_shape']}")
    print(f"  Max absolute diff: {result['max_diff']}")
    assert result["passed"], (
        f"FAILED: max_diff={result['max_diff']}, "
        f"shapes nki={result['nki_shape']} gold={result['gold_shape']}"
    )
    print("  PASSED")


def test_gemm():
    _run_and_assert("gemm")

def test_gemm_batched():
    _run_and_assert("gemm_batched")

def test_gemm_3d():
    _run_and_assert("gemm_3d")

def test_gemm_scale_residual():
    _run_and_assert("gemm_scale_residual")

def test_gemm_swish_scaling():
    _run_and_assert("gemm_swish_scaling")

def test_activation_relu():
    _run_and_assert("activation", "relu_small")

def test_activation_gelu():
    _run_and_assert("activation", "gelu_small")

def test_activation_sigmoid():
    _run_and_assert("activation", "sigmoid_small")

def test_activation_swish():
    _run_and_assert("activation", "swish_small")

def test_activation_softmax():
    _run_and_assert("activation", "softmax_small")

def test_layernorm():
    _run_and_assert("layernorm")

def test_rmsnorm():
    _run_and_assert("rmsnorm")

def test_sdpa():
    _run_and_assert("sdpa")

def test_minigpt_block():
    _run_and_assert("minigpt_block")

def test_moe():
    _run_and_assert("moe")

def test_mlp():
    _run_and_assert("mlp")

def test_mlp_shallow_wide():
    _run_and_assert("mlp_shallow_wide")

def test_mamba2():
    _run_and_assert("mamba2")
