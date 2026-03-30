#!/usr/bin/env python3
"""Standalone cycle-accurate verification of STeP baselines against compute_gold().

Usage:
    # Verify a single baseline with a specific preset:
    python verify_baseline.py gemm --preset small

    # Verify a single baseline with custom dims:
    python verify_baseline.py gemm --dims '{"M": 256, "K": 256, "N": 256}'

    # Verify all baselines that exist (all presets):
    python verify_baseline.py --all
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile

import torch
import yaml

# Ensure step_artifact is importable (mirrors PYTHONPATH from env.sh)
STEP_SRC = os.environ.get(
    "STEP_ARTIFACT_SRC",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../step_artifact/src")),
)
for subdir in ["proto", "sim", "step_py", ""]:
    p = os.path.join(STEP_SRC, subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from sim import simulate, HBMConfig, SimConfig
from utils.gold_checking import reconstruct_numpy

RTOL = 1e-3
ATOL = 1e-3

DEFAULT_HBM_CONFIG = {
    "addr_offset": 64,
    "channel_num": 32,
    "per_channel_latency": 2,
    "per_channel_init_interval": 2,
    "per_channel_outstanding": 1,
    "per_channel_start_up_time": 14,
}

DEFAULT_SIM_CONFIG = {
    "channel_depth": 2,
    "functional_sim": True,
    "mock_bf16": False,
}


def load_module(path, name="mod"):
    """Import a Python file as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_baseline(problem_path, baseline_path, dims):
    """Run cycle-accurate sim on a baseline and compare to compute_gold().

    Returns a dict with keys: correct, cycles, duration_ms, max_diff, error (if any).
    """
    problem_mod = load_module(problem_path, "problem_mod")
    baseline_mod = load_module(baseline_path, "baseline_mod")

    assert hasattr(problem_mod, "compute_gold"), f"No compute_gold() in {problem_path}"
    assert hasattr(baseline_mod, "build_graph"), f"No build_graph() in {baseline_path}"

    graph, output_op = baseline_mod.build_graph(dims)

    hbm = HBMConfig(**DEFAULT_HBM_CONFIG)
    sim = SimConfig(
        channel_depth=DEFAULT_SIM_CONFIG["channel_depth"],
        functional_sim=DEFAULT_SIM_CONFIG["functional_sim"],
        mock_bf16=DEFAULT_SIM_CONFIG["mock_bf16"],
    )

    orig_dir = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="verify_baseline_")
    pb_path = os.path.join(tmpdir, "graph.pb")

    os.chdir(tmpdir)
    cycles, duration_ms, duration_s = simulate(
        graph,
        logging=False,
        hbm_config=hbm,
        sim_config=sim,
        protobuf_file=pb_path,
        db_name=None,
    )

    result = {"cycles": cycles, "duration_ms": duration_ms, "duration_s": duration_s}

    store_name = output_op.store_file_name
    assert os.path.exists(f"{store_name}.json") and os.path.exists(f"{store_name}.npy"), (
        f"Simulation did not produce output files in {tmpdir} "
        f"(expected {store_name}.json and {store_name}.npy)"
    )

    sim_output = reconstruct_numpy(store_name, delete_npy=False)
    os.chdir(orig_dir)

    sim_tensor = torch.from_numpy(sim_output).float()
    gold = problem_mod.compute_gold(dims).float()

    assert sim_tensor.numel() == gold.numel(), (
        f"Element count mismatch: sim={sim_tensor.numel()} gold={gold.numel()} "
        f"(sim shape={tuple(sim_tensor.shape)}, gold shape={tuple(gold.shape)})"
    )

    # Pad sim to match gold's ndim (e.g. (64, 256) vs (1, 1, 64, 256)),
    # then reshape to gold's shape so element ordering is preserved.
    while sim_tensor.ndim < gold.ndim:
        sim_tensor = sim_tensor.unsqueeze(0)
    sim_tensor = sim_tensor.reshape(gold.shape)

    max_diff = (sim_tensor - gold).abs().max().item()
    passed = torch.allclose(sim_tensor, gold, rtol=RTOL, atol=ATOL)

    result["correct"] = passed
    result["max_diff"] = max_diff
    if not passed:
        result["error"] = f"Max abs diff: {max_diff}"

    return result


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "bench_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single(name, preset=None, dims=None):
    """Verify a single benchmark by name."""
    config = load_config()
    assert name in config, f"Unknown benchmark: {name}. Available: {list(config.keys())}"

    bench = config[name]
    assert bench.get("baseline"), f"No baseline defined for {name}"

    base_dir = os.path.dirname(__file__)
    problem_path = os.path.join(base_dir, bench["problem"])
    baseline_path = os.path.join(base_dir, bench["baseline"])

    if dims is None:
        assert preset, f"Must specify --preset or --dims. Available presets: {list(bench['presets'].keys())}"
        assert preset in bench["presets"], (
            f"Unknown preset '{preset}' for {name}. Available: {list(bench['presets'].keys())}"
        )
        dims = bench["presets"][preset]

    print(f"[{name}] dims={dims}")
    result = verify_baseline(problem_path, baseline_path, dims)
    status = "PASS" if result["correct"] else "FAIL"
    print(f"[{name}] {status} | cycles={result['cycles']} | max_diff={result.get('max_diff', 'N/A')}")
    if not result["correct"]:
        print(f"[{name}] Error: {result.get('error', 'unknown')}")
    return result


def run_all():
    """Verify all baselines across all presets."""
    config = load_config()
    results = {}
    for name, bench in config.items():
        if not bench.get("baseline"):
            continue
        for preset_name, dims in bench["presets"].items():
            key = f"{name}/{preset_name}"
            print(f"\n--- {key} ---")
            try:
                results[key] = run_single(name, preset=preset_name)
            except Exception as e:
                print(f"[{key}] ERROR: {e}")
                results[key] = {"correct": False, "error": str(e)}

    print("\n=== Summary ===")
    for key, r in results.items():
        status = "PASS" if r.get("correct") else "FAIL"
        print(f"  {key}: {status}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify STeP baselines against compute_gold()")
    parser.add_argument("name", nargs="?", help="Benchmark name (e.g. gemm, sdpa)")
    parser.add_argument("--preset", help="Preset name from bench_config.yaml")
    parser.add_argument("--dims", help="JSON dict of dimensions (overrides preset)")
    parser.add_argument("--all", action="store_true", help="Verify all baselines with all presets")
    args = parser.parse_args()

    assert args.name or args.all, "Specify a benchmark name or --all"

    if args.all:
        run_all()
    else:
        dims = json.loads(args.dims) if args.dims else None
        run_single(args.name, preset=args.preset, dims=dims)


if __name__ == "__main__":
    main()
