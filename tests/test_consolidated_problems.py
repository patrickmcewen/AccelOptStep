import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_gemm_get_inputs_shapes():
    from StepBench.problems.gemm import get_inputs

    dims = {"M": 64, "K": 128, "N": 256}
    inputs = get_inputs(dims)
    assert len(inputs) == 2
    assert inputs[0].shape == (64, 128)
    assert inputs[1].shape == (128, 256)


def test_gemm_compute_gold_shape():
    from StepBench.problems.gemm import compute_gold

    dims = {"M": 64, "K": 128, "N": 256}
    gold = compute_gold(dims)
    assert gold.shape == (64, 256)


def test_gemm_compute_gold_deterministic():
    from StepBench.problems.gemm import compute_gold

    dims = {"M": 32, "K": 32, "N": 32}
    g1 = compute_gold(dims)
    g2 = compute_gold(dims)
    assert torch.allclose(g1, g2)


def test_activation_relu():
    from StepBench.problems.activation import compute_gold

    dims = {"batch_size": 4, "dim": 8, "fn": "relu"}
    gold = compute_gold(dims)
    assert gold.shape == (4, 8)
    assert (gold >= 0).all()


def test_activation_all_fns():
    from StepBench.problems.activation import compute_gold

    for fn in ["relu", "gelu", "sigmoid", "swish", "softmax"]:
        dims = {"batch_size": 4, "dim": 8, "fn": fn}
        gold = compute_gold(dims)
        assert gold.shape == (4, 8), f"{fn} failed"


def test_activation_get_init_inputs():
    from StepBench.problems.activation import get_init_inputs

    dims = {"batch_size": 4, "dim": 8, "fn": "gelu"}
    init = get_init_inputs(dims)
    assert init == ["gelu"]
