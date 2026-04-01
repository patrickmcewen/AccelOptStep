"""Phase state persistence for correctness-first flow."""

import json
from pathlib import Path


def load_phase_state(exp_base_dir: Path) -> dict:
    """Load phase_state.json, returning empty dict if it doesn't exist."""
    path = exp_base_dir / "phase_state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_phase_state(exp_base_dir: Path, state: dict) -> None:
    """Write phase_state.json atomically."""
    path = exp_base_dir / "phase_state.json"
    path.write_text(json.dumps(state, indent=2))


def get_phase(state: dict, service_name: str) -> str:
    """Get current phase for a service. Defaults to 'correctness'."""
    return state.get(service_name, {}).get("phase", "correctness")


def get_correct_kernel_count(state: dict, service_name: str) -> int:
    """Get the number of correct kernels found for a service."""
    return state.get(service_name, {}).get("correct_kernel_count", 0)


def init_service_state(state: dict, service_name: str) -> dict:
    """Initialize state for a service if not present."""
    if service_name not in state:
        state[service_name] = {
            "phase": "correctness",
            "correct_kernel_count": 0,
            "correct_kernels": [],
        }
    return state


def record_correct_kernel(state: dict, service_name: str, kernel_info: dict) -> dict:
    """Record a correct kernel and return updated state."""
    state = init_service_state(state, service_name)
    state[service_name]["correct_kernels"].append(kernel_info)
    state[service_name]["correct_kernel_count"] = len(state[service_name]["correct_kernels"])
    return state


def check_transition(state: dict, service_name: str, threshold: int) -> bool:
    """Check if a service should transition to optimization phase.

    Returns True if the phase was changed.
    """
    state = init_service_state(state, service_name)
    if state[service_name]["phase"] != "correctness":
        return False
    if state[service_name]["correct_kernel_count"] >= threshold:
        state[service_name]["phase"] = "optimization"
        return True
    return False


def get_best_correct_kernel(state: dict, service_name: str) -> dict | None:
    """Return the correct kernel with the lowest cycle count, or None."""
    kernels = state.get(service_name, {}).get("correct_kernels", [])
    if not kernels:
        return None
    return min(kernels, key=lambda k: k.get("cycles", float("inf")))
