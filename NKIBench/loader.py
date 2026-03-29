"""Thin loader for NKIBench/bench_config.yaml — single entry point for NKI benchmark config."""

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
