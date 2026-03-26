import sys
import os
sys.path.insert(0, os.environ.get("STEP_ARTIFACT_SRC", "/home/ubuntu/patrick/AbstractOpt/step_artifact/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_gemm_square_builds_graph():
    from StepBench.baselines.gemm_square import build_graph
    graph, output_op = build_graph()
    assert graph is not None
    assert output_op is not None
    assert len(graph.nodes()) > 0

def test_gemm_silu_builds_graph():
    from StepBench.baselines.gemm_silu import build_graph
    graph, output_op = build_graph()
    assert graph is not None
    assert len(graph.nodes()) > 0

def test_sdpa_builds_graph():
    from StepBench.baselines.sdpa import build_graph
    graph, output_op = build_graph()
    assert graph is not None
    assert len(graph.nodes()) > 0
