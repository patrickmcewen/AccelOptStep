import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_load_config_returns_all_benchmarks():
    from StepBench.loader import load_config

    config = load_config()
    assert "gemm" in config
    assert "sdpa" in config
    assert "activation" in config
    assert len(config) == 17


def test_get_dims_returns_correct_values():
    from StepBench.loader import get_dims

    dims = get_dims("gemm", "square")
    assert dims == {"M": 4096, "K": 4096, "N": 4096}


def test_get_dims_activation_includes_fn():
    from StepBench.loader import get_dims

    dims = get_dims("activation", "relu_small")
    assert dims["fn"] == "relu"
    assert dims["batch_size"] == 256


def test_list_presets():
    from StepBench.loader import list_presets

    presets = list_presets("gemm")
    assert "small" in presets
    assert "square" in presets
    assert "large_k" in presets


def test_list_benchmarks():
    from StepBench.loader import list_benchmarks

    benchmarks = list_benchmarks()
    assert "gemm" in benchmarks
    assert "sdpa" in benchmarks
    assert len(benchmarks) == 17
