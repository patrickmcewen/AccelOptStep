"""Initialize an experiment directory for a new iteration."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(
    exp_date: str,
    last_exp_date: str,
    exp_base_dir: Path,
    project_name: str,
    org_name: str,
    logfire_enabled: bool = True,
    log_file: Path = None,
) -> None:
    exp_dir = exp_base_dir / exp_date
    last_exp_executor_results_path = exp_base_dir / last_exp_date / "executor_results.json"

    assert exp_dir.exists(), f"Experiment directory does not exist: {exp_dir}"
    assert last_exp_executor_results_path.exists(), (
        f"Last experiment executor results not found: {last_exp_executor_results_path}"
    )

    candidates_dir = exp_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        str(last_exp_executor_results_path),
        str(candidates_dir / "last_iteration_executor_results.json"),
    )

    logfire_env_name = exp_date.replace("-", "")
    if logfire_enabled:
        subprocess.run(
            ["logfire", "projects", "use", project_name, "--org", org_name],
            cwd=str(exp_dir),
            check=True,
        )
    (exp_dir / "logfire_env_name.txt").write_text(logfire_env_name)
    os.environ["LOGFIRE_ENVIRONMENT"] = logfire_env_name
    print(f"LOGFIRE_ENVIRONMENT: {logfire_env_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize experiment directory")
    parser.add_argument("exp_date", type=str)
    parser.add_argument("last_exp_date", type=str)
    parser.add_argument("exp_base_dir", type=Path)
    parser.add_argument("project_name", type=str)
    parser.add_argument("org_name", type=str)
    parser.add_argument("log_file", type=Path)
    args = parser.parse_args()

    run(
        exp_date=args.exp_date,
        last_exp_date=args.last_exp_date,
        exp_base_dir=args.exp_base_dir,
        project_name=args.project_name,
        org_name=args.org_name,
        log_file=args.log_file,
    )
