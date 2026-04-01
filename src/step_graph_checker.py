"""Programmatic constraint checker for STeP graphs.

Inspects a built STeP graph for constraint violations BEFORE expensive
profiling/simulation.  Returns a list of violation dicts (empty == all pass).
"""

from __future__ import annotations

import sys, os

# Ensure step_py is importable
_step_src = os.environ.get(
    "STEP_ARTIFACT_SRC",
    os.path.join(os.path.dirname(__file__), "..", "step_artifact", "src"),
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
    get_stream,
)
from step_py.datatype import Buffer, Tile, DynTile
from step_py.dyndim import DynDim
from step_py.functions.map_fn import Matmul, DynMatmul
from step_py.functions.map_accum_fn import Matmul as MatmulAccum


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


def _check_binarymap_shape_mismatch(graph: MultiDiGraph) -> list[dict]:
    """For BinaryMap/BinaryMapAccum nodes, verify both input streams have matching shapes."""
    violations: list[dict] = []
    for node in graph.nodes:
        if not isinstance(node, (BinaryMap, BinaryMapAccum)):
            continue

        in1_stream = get_stream(node.in1)
        in2_stream = get_stream(node.in2)

        if len(in1_stream.shape) != len(in2_stream.shape):
            violations.append({
                "check": "binarymap_shape_mismatch",
                "message": (
                    f"{type(node).__name__} {node.instance_id}: input ranks differ — "
                    f"{len(in1_stream.shape)} vs {len(in2_stream.shape)}"
                ),
                "details": {
                    "node_id": node.instance_id,
                    "node_type": type(node).__name__,
                    "in1_shape": str(in1_stream.shape),
                    "in2_shape": str(in2_stream.shape),
                },
            })
            continue

        for dim_idx, (d1, d2) in enumerate(zip(in1_stream.shape, in2_stream.shape)):
            match = True
            if isinstance(d1, DynDim) and isinstance(d2, DynDim):
                match = d1.expr.equals(d2.expr)
            else:
                match = d1 == d2
            if not match:
                violations.append({
                    "check": "binarymap_shape_mismatch",
                    "message": (
                        f"{type(node).__name__} {node.instance_id}: "
                        f"input shapes differ at dim {dim_idx} — {d1} vs {d2}"
                    ),
                    "details": {
                        "node_id": node.instance_id,
                        "node_type": type(node).__name__,
                        "dim_idx": dim_idx,
                        "in1_shape": str(in1_stream.shape),
                        "in2_shape": str(in2_stream.shape),
                    },
                })
                break  # one mismatch per node is enough
    return violations


def _check_binarymap_tile_shape_mismatch(graph: MultiDiGraph) -> list[dict]:
    """For element-wise BinaryMap/BinaryMapAccum nodes, verify input tile shapes are identical.

    The Python-side map_fn (Mul, Add, etc.) accepts broadcastable tile shapes
    (e.g. Tile(B,I) vs Tile(B,1)), but the Rust simulator requires exact match.
    Matmul ops are excluded — they inherently take different tile shapes ([M,K] x [K,N]).
    """
    violations: list[dict] = []
    # Matmul fns expect different tile shapes by design; skip them
    _matmul_fn_types = (Matmul, DynMatmul, MatmulAccum)

    for node in graph.nodes:
        if not isinstance(node, (BinaryMap, BinaryMapAccum)):
            continue

        # Skip matmul ops — tile shapes are expected to differ ([M,K] x [K,N])
        if isinstance(node.fn, _matmul_fn_types):
            continue

        in1_stream = get_stream(node.in1)
        in2_stream = get_stream(node.in2)

        t1 = in1_stream.stream_dtype
        t2 = in2_stream.stream_dtype

        # Only check Tile/DynTile pairs — Buffer/Select inputs are different codepaths
        if not (isinstance(t1, (Tile, DynTile)) and isinstance(t2, (Tile, DynTile))):
            continue

        if t1.shape != t2.shape:
            violations.append({
                "check": "binarymap_tile_shape_mismatch",
                "message": (
                    f"{type(node).__name__} {node.instance_id}: "
                    f"input tile shapes differ — {t1.shape} vs {t2.shape}. "
                    f"The simulator does not support broadcasting; tiles must match exactly."
                ),
                "details": {
                    "node_id": node.instance_id,
                    "node_type": type(node).__name__,
                    "in1_tile_shape": str(t1.shape),
                    "in2_tile_shape": str(t2.shape),
                },
            })
    return violations
