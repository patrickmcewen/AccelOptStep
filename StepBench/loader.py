"""Thin loader for bench_config.yaml — single entry point for benchmark config."""

import importlib
from pathlib import Path

import yaml

BENCH_CONFIG_PATH = Path(__file__).parent / "bench_config.yaml"


def load_config():
    """Return the full config dict from bench_config.yaml."""
    with open(BENCH_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_dims(bench_name, preset):
    """Return the dims dict for a benchmark + preset name."""
    config = load_config()
    return config[bench_name]["presets"][preset]


def list_presets(bench_name):
    """Return list of preset names for a benchmark."""
    config = load_config()
    return list(config[bench_name]["presets"].keys())


def list_benchmarks():
    """Return list of all benchmark names."""
    config = load_config()
    return list(config.keys())


def load_problem(bench_name):
    """Import and return the problem module."""
    config = load_config()
    problem_path = config[bench_name]["problem"]
    module_name = problem_path.replace("/", ".").removesuffix(".py")
    return importlib.import_module(f"StepBench.{module_name}")


def load_baseline(bench_name):
    """Import and return the baseline module, or None if no baseline exists."""
    config = load_config()
    baseline_path = config[bench_name].get("baseline")
    if not baseline_path:
        return None
    module_name = baseline_path.replace("/", ".").removesuffix(".py")
    return importlib.import_module(f"StepBench.{module_name}")
