"""Run rewrites selection, candidate selection, and profiling."""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def run(
    exp_date: str,
    exp_base_dir: Path,
    profile_mode: str,
    max_threshold: float,
    min_threshold: float,
    topk: int,
    topk_candidates: int,
    project_name: str,
    rel_tol: float,
    org_name: str,
    logfire_enabled: bool = True,
    log_file: Path = None,
    machine_config_path: str = None,
    machine_config_preset: str = "default",
    include_baseline: bool = False,
) -> None:
    base_dir = Path(os.environ["ACCELOPT_BASE_DIR"])
    prompts_base = base_dir / "prompts"
    exp_dir = exp_base_dir / exp_date

    assert exp_dir.exists(), f"Experiment directory does not exist: {exp_dir}"

    # Logfire setup
    if logfire_enabled:
        result = subprocess.run(
            ["logfire", "projects", "use", project_name, "--org", org_name],
            cwd=str(exp_dir),
        )
        if result.returncode != 0:
            print(f"WARNING: logfire setup failed (exit {result.returncode}), continuing without telemetry")
    logfire_env_name = (exp_dir / "logfire_env_name.txt").read_text().strip()
    os.environ["LOGFIRE_ENVIRONMENT"] = logfire_env_name
    print(f"LOGFIRE_ENVIRONMENT: {logfire_env_name}")

    # Rewrites selection
    start_time = _timestamp()

    (exp_dir / "rewrites").mkdir(parents=True, exist_ok=True)
    rewrites_selection_exec = base_dir / "src" / "agents" / "rewrites_selection.py"
    executor_results_path = exp_dir / "candidates" / "last_iteration_executor_results.json"
    base_prompt_path = prompts_base / "summarizer_prompts" / "base_prompt.txt"
    user_template_path = prompts_base / "summarizer_prompts" / "user_prompt_template.txt"
    output_list_path = exp_dir / "rewrites" / "rewrites_selection_output_list.json"
    output_speedups_path = exp_dir / "rewrites" / "rewrites_selection_output_speedups.json"
    output_plan_ids_path = exp_dir / "rewrites" / "rewrites_selection_output_plan_ids.json"
    model_config_path_summarizer = exp_base_dir / "configs" / "summarizer_config.json"
    subprocess.run(
        [
            sys.executable, str(rewrites_selection_exec),
            "--executor_results_path", str(executor_results_path),
            "--base_prompt_path", str(base_prompt_path),
            "--user_template_path", str(user_template_path),
            "--output_list_path", str(output_list_path),
            "--max_threshold", str(max_threshold),
            "--min_threshold", str(min_threshold),
            "--topk", str(topk),
            "--output_plan_ids_path", str(output_plan_ids_path),
            "--output_speedups_path", str(output_speedups_path),
            "--model_config_path", str(model_config_path_summarizer),
            "--log_file", str(log_file),
        ],
        cwd=str(exp_dir),
        check=True,
    )

    end_time = _timestamp()
    with open(exp_dir / "rewrites_selection_start_end_time.txt", "a") as f:
        f.write(f"{start_time},{end_time}\n")

    # Select candidates
    start_time = _timestamp()

    select_candidates_exec = base_dir / "src" / "agents" / "select_candidates.py"
    output_base_path = exp_dir / "candidates"
    select_cmd = [
        sys.executable, str(select_candidates_exec),
        "--executor_results_path", str(executor_results_path),
        "--output_base_path", str(output_base_path),
        "--topk", str(topk_candidates),
        "--log_file", str(log_file),
    ]
    subprocess.run(select_cmd, cwd=str(exp_dir), check=True)

    end_time = _timestamp()
    with open(exp_dir / "select_candidates_start_end_time.txt", "a") as f:
        f.write(f"{start_time},{end_time}\n")

    # Profile candidates
    start_time = _timestamp()

    candidates_path = exp_dir / "candidates" / "candidates.csv"
    profile_output_path = exp_dir / "candidates" / "profile_results.csv"

    if candidates_path.stat().st_size == 0:
        print("WARNING: candidates.csv is empty (0 optimization items produced), skipping profiling")
        profile_output_path.write_text("")
    else:
        sequential_profile_exec = base_dir / "src" / "agents" / "sequential_profile.py"
        subprocess.run(
            [
                sys.executable, str(sequential_profile_exec),
                "--candidates_path", str(candidates_path),
                "--output_path", str(profile_output_path),
                "--profile_mode", profile_mode,
                "--rel_tol", str(rel_tol),
                "--log_file", str(log_file),
            ],
            cwd=str(exp_dir),
            check=True,
        )

    end_time = _timestamp()
    with open(exp_dir / "profile_candidates_start_end_time.txt", "a") as f:
        f.write(f"{start_time},{end_time}\n")
