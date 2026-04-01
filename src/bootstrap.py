"""Environment bootstrap — Docker/STeP cache, PYTHONPATH setup."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

STEP_DOCKER_IMAGE = "step_artifact"
STEP_CACHE_DIR = Path.home() / ".cache" / "acceloptstep" / "step_artifact_src"
STEP_SRC_IN_CONTAINER = "/root/step_artifact/src"
STEP_PERF_IN_CONTAINER = "/root/step_artifact/step-perf"


def load_config(config_path: str, preset: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    assert preset in raw, f"Preset '{preset}' not found in {config_path}. Available: {[k for k in raw if k != 'defaults']}"
    preset_cfg = raw[preset] or {}
    return {**defaults, **preset_cfg}


def _find_or_start_step_container() -> str:
    """Find a running step_artifact container or start one. Returns container ID."""
    result = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"ancestor={STEP_DOCKER_IMAGE}"],
        capture_output=True, text=True,
    )
    container_id = result.stdout.strip().split("\n")[0]
    if container_id:
        return container_id

    print(">>> No running step_artifact container found. Starting one...")
    result = subprocess.run(
        ["docker", "run", "-dit", STEP_DOCKER_IMAGE, "bash"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _ensure_step_artifact_cached() -> Path:
    """Ensure STeP artifact sources are cached on the host. Returns the cache path."""
    marker = STEP_CACHE_DIR / ".cached"
    if marker.exists():
        return STEP_CACHE_DIR

    print(">>> Caching STeP artifact sources from Docker container...")
    container_id = _find_or_start_step_container()

    # Remove stale cache if any
    if STEP_CACHE_DIR.exists():
        shutil.rmtree(STEP_CACHE_DIR)

    # docker cp extracts into parent dir: cp container:/root/step_artifact/src -> cache_dir
    STEP_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "cp", f"{container_id}:{STEP_SRC_IN_CONTAINER}", str(STEP_CACHE_DIR)],
        check=True,
    )
    marker.touch()
    print(f"    Cached to {STEP_CACHE_DIR}")
    return STEP_CACHE_DIR


def _ensure_step_perf_installed(python: str) -> None:
    """Ensure step_perf Rust simulator is installed for the given Python interpreter."""
    result = subprocess.run(
        [python, "-c", "import step_perf"],
        capture_output=True,
    )
    if result.returncode == 0:
        return

    print(">>> Building step_perf for host Python (one-time)...")
    container_id = _find_or_start_step_container()

    build_dir = Path("/tmp/step-perf-build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    subprocess.run(
        ["docker", "cp", f"{container_id}:{STEP_PERF_IN_CONTAINER}", str(build_dir)],
        check=True,
    )
    subprocess.run(
        [python, "-m", "pip", "install", "maturin[patchelf]"],
        capture_output=True, check=True,
    )
    subprocess.run(
        [str(Path(python).parent / "maturin"), "build", "--release", "--interpreter", python],
        cwd=str(build_dir), capture_output=True, check=True,
    )
    wheels = list((build_dir / "target" / "wheels").glob("step_perf-*.whl"))
    assert wheels, "step_perf wheel build produced no output"
    subprocess.run(
        [python, "-m", "pip", "install", str(wheels[0])],
        capture_output=True, check=True,
    )
    print("    step_perf installed.")


def setup_environment(script_dir: Path, cfg: dict):
    """Set ACCELOPT_BASE_DIR, PYTHONPATH, and ensure STeP dependencies are available."""
    os.environ["ACCELOPT_BASE_DIR"] = str(script_dir)

    if not cfg.get("logfire_enabled", True):
        os.environ["LOGFIRE_SEND_TO_LOGFIRE"] = "false"

    # Determine step_artifact_src path
    step_artifact_src = cfg.get("step_artifact_src")
    if step_artifact_src is None:
        step_artifact_src = str(_ensure_step_artifact_cached())
        _ensure_step_perf_installed(sys.executable)

    existing = os.environ.get("PYTHONPATH", "")
    additions = [str(script_dir), step_artifact_src, f"{step_artifact_src}/step_py", f"{step_artifact_src}/sim", f"{step_artifact_src}/proto"]
    os.environ["PYTHONPATH"] = ":".join(additions + ([existing] if existing else []))
    # Also add to sys.path so imports work in the current process
    for p in reversed(additions):
        if p not in sys.path:
            sys.path.insert(0, p)
