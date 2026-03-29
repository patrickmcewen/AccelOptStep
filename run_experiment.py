#!/usr/bin/env python3
"""AccelOptStep — Run the agentic optimization loop.

Usage:
    python run_experiment.py --config config.yaml --preset full_run

This script:
    1. Sets up environment (ACCELOPT_BASE_DIR, PYTHONPATH)
    2. Generates profile_results.csv from baselines (if needed)
    3. Scaffolds experiment directories via inline logic
    4. Launches the beam search loop for each problem

Prerequisites:
    - vLLM server running at http://localhost:31001/v1 (or update configs/)
    - STeP runtime: step_artifact/src on PYTHONPATH (auto-configured below)
    - For cycle_accurate mode: Docker container with step_artifact
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

# Ensure AccelOptStep is importable for templates.complete_local package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from templates.complete_local import run_single_loop

LA = ZoneInfo("America/Los_Angeles")


def load_config(config_path: str, preset: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    assert preset in raw, f"Preset '{preset}' not found in {config_path}. Available: {[k for k in raw if k != 'defaults']}"
    preset_cfg = raw[preset] or {}
    return {**defaults, **preset_cfg}


def setup_environment(script_dir: Path, cfg: dict):
    """Set ACCELOPT_BASE_DIR and PYTHONPATH."""
    os.environ["ACCELOPT_BASE_DIR"] = str(script_dir)

    if not cfg.get("logfire_enabled", True):
        os.environ["LOGFIRE_SEND_TO_LOGFIRE"] = "false"

    step_artifact_src = cfg.get("step_artifact_src")
    if step_artifact_src is None:
        step_artifact_src = str((script_dir / ".." / "step_artifact" / "src").resolve())

    existing = os.environ.get("PYTHONPATH", "")
    additions = [step_artifact_src, f"{step_artifact_src}/step_py", f"{step_artifact_src}/sim", f"{step_artifact_src}/proto"]
    os.environ["PYTHONPATH"] = ":".join(additions + ([existing] if existing else []))


def generate_profile_csv(script_dir: Path, experiments_dir: Path, profile_csv: Path, profile_mode: str, cfg: dict,
                         machine_config_path: str | None = None, machine_config_preset: str = "default",
                         pipeline: str = "pytorch-step"):
    """Step 1: Generate profile_results.csv if it's empty (header-only)."""

    print(f">>> Generating profile_results.csv (profiling baselines with mode={profile_mode})...")
    cmd = [
        sys.executable,
        str(script_dir / "scripts" / "collect_candidates.py"),
        "--output_candidates_path", str(experiments_dir / "candidates.csv"),
        "--output_profile_path", str(profile_csv),
        "--profile_mode", profile_mode,
        "--mode", "construct",
    ]
    if machine_config_path:
        cmd += ["--machine_config_path", machine_config_path]
    cmd += ["--machine_config_preset", machine_config_preset]
    cmd += ["--pipeline", pipeline]
    benchmarks = cfg.get("benchmarks", "all")
    presets = cfg.get("presets", "all")
    if isinstance(benchmarks, list):
        cmd += ["--benchmarks", ",".join(benchmarks)]
    elif benchmarks != "all":
        cmd += ["--benchmarks", str(benchmarks)]
    if isinstance(presets, list):
        cmd += ["--presets", ",".join(presets)]
    elif presets != "all":
        cmd += ["--presets", str(presets)]
    subprocess.run(cmd, check=True)
    new_count = sum(1 for _ in profile_csv.open()) - 1
    print(f">>> Done. {new_count} baselines profiled.")


def scaffold_experiments(script_dir: Path, experiments_dir: Path, cfg: dict, exp_date_base: str,
                         machine_config_path: str | None = None, machine_config_preset: str = "default",
                         pipeline: str = "pytorch-step", prompts_dir: str | None = None):
    """Step 2: Scaffold experiment directories."""
    iters = cfg["iters"]
    breadth = cfg["breadth"]
    num_samples = cfg["num_samples"]
    topk_candidates = cfg["topk_candidates"]
    topk = cfg["topk"]
    exp_n = cfg["exp_n"]
    max_threshold = cfg["max_threshold"]
    min_threshold = cfg["min_threshold"]
    rel_tol = cfg["rel_tol"]
    profile_mode = cfg["profile_mode"]
    project_name = cfg["project_name"]
    org_name = cfg["org_name"]
    logfire_enabled = cfg.get("logfire_enabled", True)

    print()
    print(">>> Scaffolding experiment directories...")
    print(f"    EXP_DATE_BASE={exp_date_base}")
    print(f"    PROFILE_MODE={profile_mode}")
    print(f"    ITERS={iters}  BREADTH={breadth}  NUM_SAMPLES={num_samples}")
    print(f"    TOPK_CANDIDATES={topk_candidates}  TOPK={topk}  EXP_N={exp_n}")
    print()

    exp_base_dir = (experiments_dir / ".." / "checkpoints" / exp_date_base).resolve()
    assert not exp_base_dir.exists(), (
        f"Checkpoint directory already exists: {exp_base_dir}\n"
        f"Choose a different exp_date_base or remove the old directory."
    )

    proxy_problem_list_df = pd.read_csv(experiments_dir / "candidates.csv")
    proxy_profile_results_df = pd.read_csv(experiments_dir / "profile_results.csv")

    first_exp_date = datetime.now(LA).strftime("%m-%d-%H-%M")

    problem_configs = []

    for index, row in proxy_problem_list_df.iterrows():
        service_name = row["service_name"]
        new_exp_base_dir = exp_base_dir / service_name
        os.makedirs(new_exp_base_dir, exist_ok=False)
        shutil.copytree(experiments_dir / "configs", new_exp_base_dir / "configs", dirs_exist_ok=True)

        eval_prefix = f"eval-{index}-{exp_date_base}"
        eval_first_exp_date = f"{eval_prefix}-{first_exp_date}"
        new_exp_candidates_dir = new_exp_base_dir / eval_first_exp_date / "candidates"
        os.makedirs(new_exp_candidates_dir, exist_ok=False)

        profile_results_df = proxy_profile_results_df[proxy_profile_results_df["service_name"] == service_name]
        profile_results_df.to_csv(new_exp_candidates_dir / "profile_results.csv", index=False)

        debug_log_path = new_exp_base_dir / "debug.log"
        loop_kwargs = dict(
            exp_base_dir=new_exp_base_dir,
            init_exp_date=eval_first_exp_date,
            exp_date_prefix=eval_prefix,
            profile_mode=profile_mode,
            project_name=project_name,
            rel_tol=rel_tol,
            org_name=org_name,
            logfire_enabled=logfire_enabled,
            iters=iters,
            breadth=breadth,
            topk_candidates=topk_candidates,
            num_samples=num_samples,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            topk=topk,
            exp_n=exp_n,
            log_file=debug_log_path,
            machine_config_path=machine_config_path,
            machine_config_preset=machine_config_preset,
            pipeline=pipeline,
            prompts_dir=prompts_dir,
        )
        problem_configs.append((service_name, loop_kwargs))

        # Generate a standalone .py wrapper for manual execution
        kwargs_lines = ",\n    ".join(
            f"{k}=Path({str(v)!r})" if isinstance(v, Path) else f"{k}={v!r}"
            for k, v in loop_kwargs.items()
        )
        wrapper_content = (
            f"#!/usr/bin/env python3\n"
            f"import sys\n"
            f"from pathlib import Path\n"
            f"sys.path.insert(0, {str(script_dir)!r})\n"
            f"from templates.complete_local.run_single_loop import run\n"
            f"\n"
            f"run(\n"
            f"    {kwargs_lines},\n"
            f")\n"
        )
        wrapper_path = exp_base_dir / f"run_single_loop_{service_name}.py"
        wrapper_path.write_text(wrapper_content)

        print(f"  Created: {service_name}")
        print(f"    Dir:    {new_exp_base_dir}")
        print(f"    Script: {wrapper_path}")

    print(f"\nScaffolded {len(proxy_problem_list_df)} problems under {exp_base_dir}")
    return exp_base_dir, problem_configs


def _tee_to_log(log_path: Path, fn, *args, **kwargs):
    """Run fn() while teeing all stdout/stderr (including subprocess output) to log_path."""
    log_f = open(log_path, "w")
    read_fd, write_fd = os.pipe()
    # Increase pipe buffer to avoid BrokenPipeError from high-volume subprocess output (e.g. NKI)
    import fcntl
    F_SETPIPE_SZ = 1031
    fcntl.fcntl(write_fd, F_SETPIPE_SZ, 1048576)  # 1MB

    # Save original fds and Python streams
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr

    # Redirect OS-level fds 1 and 2 to the write end of the pipe
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    # Point Python streams at the redirected fds
    sys.stdout = os.fdopen(1, "w", closefd=False)
    sys.stderr = os.fdopen(2, "w", closefd=False)

    # Reader thread: read from pipe, write to both log and original stdout
    def _reader():
        with os.fdopen(read_fd, "r") as pipe_read:
            for line in pipe_read:
                os.write(saved_stdout_fd, line.encode())
                log_f.write(line)
                log_f.flush()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        fn(*args, **kwargs)
    finally:
        # Flush Python streams before restoring
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original fds (this closes the pipe write end on fd 1/2,
        # signaling EOF to the reader thread)
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)

        # Wait for reader thread to drain before closing saved fds it writes to
        reader.join()

        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

        sys.stdout = saved_stdout
        sys.stderr = saved_stderr

        log_f.close()


def launch_loops(checkpoint_dir: Path, problem_configs: list, dry_run: bool):
    """Step 3: Launch the optimization loops."""
    scripts = sorted(checkpoint_dir.glob("run_single_loop_*.py"))

    print()
    print(">>> Generated run scripts:")
    for s in scripts:
        print(f"    {s}")

    if dry_run:
        print()
        print(">>> dry_run=true — stopping before execution.")
        print("    To run a single problem:  python <script_path>")
        print("    To run all problems:      set dry_run: false in your config")
        return

    print()
    print(">>> Launching optimization loops (sequential per problem)...")
    print("    Press Ctrl+C to stop.")
    print()

    for service_name, loop_kwargs in problem_configs:
        log_path = checkpoint_dir / f"{service_name}.log"
        print(f"=== Starting: {service_name} (logging to {log_path}) ===")
        _tee_to_log(log_path, run_single_loop.run, **loop_kwargs)
        print(f"=== Finished: {service_name} ===")
        print()

    print(f">>> All problems complete. Results in: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="AccelOptStep — Run the agentic optimization loop")
    parser.add_argument("--config_file", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--config", required=True, help="Named preset from the config file (e.g. full_run, test_small_scale_run)")
    args = parser.parse_args()

    cfg = load_config(args.config_file, args.config)
    script_dir = Path(__file__).resolve().parent

    # Resolve exp_date_base (auto-generate if not in config)
    exp_date_base = cfg.get("exp_date_base") or datetime.now().strftime("%Y-%m-%d-%H%M%S")

    from pipeline_registry import resolve_pipeline

    # Support both new 'pipeline' key and legacy 'benchmark_type' for backward compat
    pipeline_str = cfg.get("pipeline") or cfg.get("benchmark_type", "pytorch-step")
    pipeline = resolve_pipeline(pipeline_str)

    prompts_dir = str(script_dir / "prompts" / pipeline["prompts_subdir"])
    if pipeline["needs_machine_config"]:
        machine_config_preset = cfg.get("machine_config", "default")
        machine_config_path = str(script_dir / pipeline["bench_dir"] / "machine_config.yaml")
    else:
        machine_config_path = None
        machine_config_preset = "default"

    setup_environment(script_dir, cfg)

    experiments_dir = script_dir / "experiments" / "full_complete_local"
    profile_csv = experiments_dir / "profile_results.csv"

    generate_profile_csv(script_dir, experiments_dir, profile_csv, cfg["profile_mode"], cfg,
                         machine_config_path=machine_config_path, machine_config_preset=machine_config_preset,
                         pipeline=pipeline_str)
    checkpoint_dir, problem_configs = scaffold_experiments(script_dir, experiments_dir, cfg, exp_date_base,
                                                          machine_config_path=machine_config_path,
                                                          machine_config_preset=machine_config_preset,
                                                          pipeline=pipeline_str,
                                                          prompts_dir=prompts_dir)
    launch_loops(checkpoint_dir, problem_configs, cfg.get("dry_run", False))


if __name__ == "__main__":
    main()
