"""Programmatic constraint checker for STeP graphs.

Inspects a built STeP graph for constraint violations BEFORE expensive
profiling/simulation.  Returns a list of violation dicts (empty == all pass).
"""

from __future__ import annotations

import sys, os

# Ensure step_py is importable
_step_src = os.environ.get(
    "STEP_ARTIFACT_SRC",
    os.path.join(os.path.dirname(__file__), "..", "..", "step_artifact", "src"),
)
if _step_src not in sys.path:
    sys.path.insert(0, _step_src)

from networkx import MultiDiGraph
from step_py.ops import (
    StepOps,
    OffChipStore,
    Broadcast,
    Streamify,
    BinaryMap,
    BinaryMapAccum,
    UnaryMap,
    Accum,
)
from step_py.datatype import Buffer


def check_step_graph(
    graph: MultiDiGraph,
    output_op: StepOps,
    total_compute_bw: int,
    on_chip_memory_bytes: int,
) -> list[dict]:
    """Return a list of violation dicts for *graph*.  Empty list means all checks pass."""

    violations: list[dict] = []

    violations.extend(_check_compute_bw_budget(graph, total_compute_bw))
    violations.extend(_check_no_offchipstore(graph))
    violations.extend(_check_broadcast_index_oob(graph))
    violations.extend(_check_streamify_non_buffer_input(graph))

    return violations


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_compute_bw_budget(graph: MultiDiGraph, total_compute_bw: int) -> list[dict]:
    """Sum compute_bw across all nodes that have it; flag if over budget."""
    total = 0
    for node in graph.nodes:
        if hasattr(node, "compute_bw"):
            total += node.compute_bw

    if total > total_compute_bw:
        return [{
            "check": "compute_bw_budget",
            "message": (
                f"Total compute_bw ({total}) exceeds budget ({total_compute_bw})"
            ),
            "details": {"total": total, "budget": total_compute_bw},
        }]
    return []


def _check_no_offchipstore(graph: MultiDiGraph) -> list[dict]:
    """Verify at least one OffChipStore node exists."""
    for node in graph.nodes:
        if isinstance(node, OffChipStore):
            return []
    return [{
        "check": "no_offchipstore",
        "message": "Graph has no OffChipStore node — output will never be written",
        "details": {},
    }]


def _check_broadcast_index_oob(graph: MultiDiGraph) -> list[dict]:
    """For edges (broadcast, i) used as inputs, verify i < num_consumers."""
    violations: list[dict] = []
    for node in graph.nodes:
        try:
            inputs = node.input_list
        except (NotImplementedError, AttributeError):
            continue
        for inp in inputs:
            if isinstance(inp, tuple) and len(inp) == 2:
                src, idx = inp
                if isinstance(src, Broadcast) and idx >= src.num_consumers:
                    violations.append({
                        "check": "broadcast_index_oob",
                        "message": (
                            f"Broadcast {src.instance_id} indexed at {idx} "
                            f"but only has {src.num_consumers} consumers"
                        ),
                        "details": {
                            "broadcast_id": src.instance_id,
                            "index": idx,
                            "num_consumers": src.num_consumers,
                        },
                    })
    return violations


def _check_streamify_non_buffer_input(graph: MultiDiGraph) -> list[dict]:
    """For Streamify nodes, verify the input stream_dtype is Buffer."""
    violations: list[dict] = []
    for node in graph.nodes:
        if not isinstance(node, Streamify):
            continue
        from step_py.ops import get_stream
        in_stream = get_stream(node.input)
        if not isinstance(in_stream.stream_dtype, Buffer):
            violations.append({
                "check": "streamify_non_buffer_input",
                "message": (
                    f"Streamify {node.instance_id} input has stream_dtype "
                    f"{type(in_stream.stream_dtype).__name__}, expected Buffer"
                ),
                "details": {
                    "streamify_id": node.instance_id,
                    "actual_dtype": type(in_stream.stream_dtype).__name__,
                },
            })
    return violations
