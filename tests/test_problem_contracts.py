"""Verify all problem files follow the dims-based contract."""
import sys
import os
import inspect
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from StepBench.loader import load_config, load_problem


def test_all_problems_have_required_exports():
    config = load_config()
    for bench_name in config:
        problem = load_problem(bench_name)
        assert hasattr(problem, "Model"), f"{bench_name}: missing Model class"
        assert hasattr(problem, "get_inputs"), f"{bench_name}: missing get_inputs"
        assert hasattr(problem, "get_init_inputs"), f"{bench_name}: missing get_init_inputs"
        assert hasattr(problem, "compute_gold"), f"{bench_name}: missing compute_gold"


def test_all_get_inputs_accept_dims():
    config = load_config()
    for bench_name in config:
        problem = load_problem(bench_name)
        sig = inspect.signature(problem.get_inputs)
        params = list(sig.parameters.keys())
        assert params == ["dims"], (
            f"{bench_name}.get_inputs signature should be (dims), got ({', '.join(params)})"
        )


def test_all_compute_gold_accept_dims():
    config = load_config()
    for bench_name in config:
        problem = load_problem(bench_name)
        sig = inspect.signature(problem.compute_gold)
        params = list(sig.parameters.keys())
        assert params == ["dims"], (
            f"{bench_name}.compute_gold signature should be (dims), got ({', '.join(params)})"
        )


def test_all_problems_run_with_small_preset():
    """Smoke test: every problem can compute_gold with its first preset."""
    config = load_config()
    for bench_name, bench_info in config.items():
        presets = list(bench_info["presets"].keys())
        small_preset = "small" if "small" in presets else presets[0]
        dims = bench_info["presets"][small_preset]

        problem = load_problem(bench_name)
        gold = problem.compute_gold(dims)
        assert isinstance(gold, torch.Tensor), f"{bench_name}: compute_gold didn't return a tensor"
        assert gold.numel() > 0, f"{bench_name}: compute_gold returned empty tensor"
