"""Scan executor results for correct kernels."""

import json
from pathlib import Path


def scan_for_correct_kernels(executor_results_path: Path, iteration: str) -> list[dict]:
    """Read executor_results.json and return info for each correct kernel.

    Returns list of dicts with keys: code, cycles, metadata, iteration, plan_id
    """
    assert executor_results_path.exists(), f"executor_results.json not found: {executor_results_path}"
    results = json.loads(executor_results_path.read_text())

    correct_kernels = []
    for record in results.get("executor_results", []):
        for key, value in record.items():
            if key in ("service_name", "case_name") or not isinstance(value, dict):
                continue
            metadata_str = value.get("kernel_metadata", "{}")
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            if metadata.get("correct", False):
                correct_kernels.append({
                    "code": value.get("body", ""),
                    "cycles": metadata.get("cycles", float("inf")),
                    "metadata": metadata,
                    "iteration": iteration,
                    "plan_id": key,
                })

    return correct_kernels
