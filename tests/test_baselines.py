import sys
import os
sys.path.insert(0, os.environ.get("STEP_ARTIFACT_SRC", "/home/ubuntu/patrick/AbstractOpt/step_artifact/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from StepBench.loader import get_dims


def test_gemm_builds_graph():
    from StepBench.baselines.gemm import build_graph
    dims = get_dims("gemm", "small")
    graph, output_op = build_graph(dims)
    assert graph is not None
    assert output_op is not None
    assert len(graph.nodes()) > 0


def test_gemm_swish_scaling_builds_graph():
    from StepBench.baselines.gemm_swish_scaling import build_graph
    dims = get_dims("gemm_swish_scaling", "small")
    graph, output_op = build_graph(dims)
    assert graph is not None
    assert len(graph.nodes()) > 0


def test_sdpa_builds_graph():
    from StepBench.baselines.sdpa import build_graph
    dims = get_dims("sdpa", "small")
    graph, output_op = build_graph(dims)
    assert graph is not None
    assert len(graph.nodes()) > 0
