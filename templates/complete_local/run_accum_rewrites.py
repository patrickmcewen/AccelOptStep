"""Run rewrites selection, candidate selection, and profiling."""

import argparse
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
) -> None:
    base_dir = Path(os.environ["ACCELOPT_BASE_DIR"])
    exp_dir = exp_base_dir / exp_date

    assert exp_dir.exists(), f"Experiment directory does not exist: {exp_dir}"

    # Logfire setup
    if logfire_enabled:
        subprocess.run(
            ["logfire", "projects", "use", project_name, "--org", org_name],
            cwd=str(exp_dir),
            check=True,
        )
    logfire_env_name = (exp_dir / "logfire_env_name.txt").read_text().strip()
    os.environ["LOGFIRE_ENVIRONMENT"] = logfire_env_name
    print(f"LOGFIRE_ENVIRONMENT: {logfire_env_name}")

    # Rewrites selection
    start_time = _timestamp()

    (exp_dir / "rewrites").mkdir(parents=True, exist_ok=True)
    rewrites_selection_exec = base_dir / "scripts" / "rewrites_selection.py"
    executor_results_path = exp_dir / "candidates" / "last_iteration_executor_results.json"
    base_prompt_path = base_dir / "prompts" / "summarizer_prompts" / "base_prompt.txt"
    user_template_path = base_dir / "prompts" / "summarizer_prompts" / "user_prompt_template.txt"
    output_list_path = exp_dir / "rewrites" / "rewrites_selection_output_list.json"
    output_speedups_path = exp_dir / "rewrites" / "rewrites_selection_output_speedups.json"
    output_plan_ids_path = exp_dir / "rewrites" / "rewrites_selection_output_plan_ids.json"
    model_config_path = exp_base_dir / "configs" / "summarizer_config.json"
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
            "--model_config_path", str(model_config_path),
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

    select_candidates_exec = base_dir / "scripts" / "select_candidates.py"
    output_base_path = exp_dir / "candidates"
    subprocess.run(
        [
            sys.executable, str(select_candidates_exec),
            "--executor_results_path", str(executor_results_path),
            "--output_base_path", str(output_base_path),
            "--topk", str(topk_candidates),
            "--log_file", str(log_file),
        ],
        cwd=str(exp_dir),
        check=True,
    )

    end_time = _timestamp()
    with open(exp_dir / "select_candidates_start_end_time.txt", "a") as f:
        f.write(f"{start_time},{end_time}\n")

    # Profile candidates
    start_time = _timestamp()

    sequential_profile_exec = base_dir / "scripts" / "sequential_profile.py"
    candidates_path = exp_dir / "candidates" / "candidates.csv"
    profile_output_path = exp_dir / "candidates" / "profile_results.csv"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rewrites selection, candidate selection, and profiling")
    parser.add_argument("exp_date", type=str)
    parser.add_argument("exp_base_dir", type=Path)
    parser.add_argument("profile_mode", type=str)
    parser.add_argument("max_threshold", type=float)
    parser.add_argument("min_threshold", type=float)
    parser.add_argument("topk", type=int)
    parser.add_argument("topk_candidates", type=int)
    parser.add_argument("project_name", type=str)
    parser.add_argument("rel_tol", type=float)
    parser.add_argument("org_name", type=str)
    parser.add_argument("log_file", type=Path)
    args = parser.parse_args()

    run(
        exp_date=args.exp_date,
        exp_base_dir=args.exp_base_dir,
        profile_mode=args.profile_mode,
        max_threshold=args.max_threshold,
        min_threshold=args.min_threshold,
        topk=args.topk,
        topk_candidates=args.topk_candidates,
        project_name=args.project_name,
        rel_tol=args.rel_tol,
        org_name=args.org_name,
        log_file=args.log_file,
    )
