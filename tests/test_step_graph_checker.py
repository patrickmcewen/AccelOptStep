"""Tests for step_graph_checker using real STeP graph ops."""

import sys
import os
import pytest
import torch

# Ensure step_py and src are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_step_src = os.path.join(os.path.dirname(__file__), "..", "..", "step_artifact", "src")
if _step_src not in sys.path:
    sys.path.insert(0, _step_src)

from networkx import MultiDiGraph
from step_py.ops import (
    OffChipLoad,
    OffChipStore,
    BinaryMap,
    Broadcast,
    Bufferize,
    Streamify,
)
from step_py.functions.map_fn import Matmul, Mul, Add
from step_py.datatype import Tile, Float16, Stream, Buffer

from src.step_graph_checker import check_step_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gemm_graph(compute_bw=4096):
    """Build a minimal correct GEMM graph: load A, load B -> matmul -> store."""
    graph = MultiDiGraph()

    # Two input tensors: A [128, 128] fp16, B [128, 128] fp16
    a_tensor = torch.randn(128, 128, dtype=torch.float16)
    b_tensor = torch.randn(128, 128, dtype=torch.float16)

    load_a = OffChipLoad(
        underlying=a_tensor,
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=128,
        tile_col=128,
        par_dispatch=1,
    )
    load_b = OffChipLoad(
        underlying=b_tensor,
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=128,
        tile_col=128,
        par_dispatch=1,
    )

    graph.add_node(load_a)
    graph.add_node(load_b)

    matmul = BinaryMap(
        graph=graph,
        in1=load_a,
        in2=load_b,
        fn=Matmul(),
        write_back_mu=False,
        compute_bw=compute_bw,
    )

    store = OffChipStore(
        graph=graph,
        input=matmul,
        par_dispatch=1,
    )

    return graph, store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clean_graph_passes():
    graph, store = _make_gemm_graph(compute_bw=4096)
    violations = check_step_graph(graph, store, total_compute_bw=4096, on_chip_memory_bytes=1024 * 1024)
    assert violations == []


def test_compute_bw_exceeded():
    graph, store = _make_gemm_graph(compute_bw=4096)
    violations = check_step_graph(graph, store, total_compute_bw=2048, on_chip_memory_bytes=1024 * 1024)
    assert len(violations) == 1
    assert violations[0]["check"] == "compute_bw_budget"
    assert violations[0]["details"]["total"] == 4096
    assert violations[0]["details"]["budget"] == 2048


def test_no_offchipstore():
    """Graph with no OffChipStore should be flagged."""
    graph = MultiDiGraph()

    a_tensor = torch.randn(128, 128, dtype=torch.float16)
    b_tensor = torch.randn(128, 128, dtype=torch.float16)

    load_a = OffChipLoad(
        underlying=a_tensor, stride=(1,), out_shape_tiled=(1,),
        tile_row=128, tile_col=128, par_dispatch=1,
    )
    load_b = OffChipLoad(
        underlying=b_tensor, stride=(1,), out_shape_tiled=(1,),
        tile_row=128, tile_col=128, par_dispatch=1,
    )

    graph.add_node(load_a)
    graph.add_node(load_b)

    matmul = BinaryMap(
        graph=graph, in1=load_a, in2=load_b,
        fn=Matmul(), write_back_mu=False, compute_bw=4096,
    )
    # No OffChipStore — the violation we want to detect

    violations = check_step_graph(graph, matmul, total_compute_bw=4096, on_chip_memory_bytes=1024 * 1024)
    checks = [v["check"] for v in violations]
    assert "no_offchipstore" in checks


def test_broadcast_with_valid_softmax():
    """A correct broadcast pattern (softmax-like) should pass broadcast_index_oob."""
    graph = MultiDiGraph()

    # Single input tensor
    x_tensor = torch.randn(128, 128, dtype=torch.float16)

    load_x = OffChipLoad(
        underlying=x_tensor, stride=(1,), out_shape_tiled=(1,),
        tile_row=128, tile_col=128, par_dispatch=1,
    )
    graph.add_node(load_x)

    # Broadcast to 2 consumers
    bcast = Broadcast(graph=graph, input=load_x, num_consumers=2)

    # Consumer 0 and consumer 1 — both valid indices
    mul_node = BinaryMap(
        graph=graph, in1=(bcast, 0), in2=(bcast, 1),
        fn=Mul(), write_back_mu=False, compute_bw=2048,
    )

    store = OffChipStore(graph=graph, input=mul_node, par_dispatch=1)

    violations = check_step_graph(graph, store, total_compute_bw=4096, on_chip_memory_bytes=1024 * 1024)
    # No broadcast_index_oob violations expected
    bcast_violations = [v for v in violations if v["check"] == "broadcast_index_oob"]
    assert bcast_violations == []


def test_compute_bw_exact_budget_passes():
    """Exact match of compute_bw to budget should pass."""
    graph, store = _make_gemm_graph(compute_bw=2048)
    violations = check_step_graph(graph, store, total_compute_bw=2048, on_chip_memory_bytes=1024 * 1024)
    bw_violations = [v for v in violations if v["check"] == "compute_bw_budget"]
    assert bw_violations == []
