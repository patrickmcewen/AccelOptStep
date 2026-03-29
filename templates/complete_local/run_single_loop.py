"""Run the full experiment loop: first iteration + N subsequent iterations."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure AccelOptStep root is importable when run standalone
_accelopt_root = Path(__file__).resolve().parent.parent.parent
if str(_accelopt_root) not in sys.path:
    sys.path.insert(0, str(_accelopt_root))

from templates.complete_local import run_accum_rewrites, run_body, run_first, run_init
from templates.complete_local.logging_config import setup_problem_logger

LA = ZoneInfo("America/Los_Angeles")


def run(
    exp_base_dir: Path,
    init_exp_date: str,
    exp_date_prefix: str,
    profile_mode: str,
    project_name: str,
    rel_tol: float,
    org_name: str,
    logfire_enabled: bool,
    iters: int,
    breadth: int,
    topk_candidates: int,
    num_samples: int,
    max_threshold: float,
    min_threshold: float,
    topk: int,
    exp_n: int,
    log_file: Path,
    machine_config_path: str | None = None,
    machine_config_preset: str = "default",
) -> None:
    setup_problem_logger(log_file)
    base_dir = Path(os.environ["ACCELOPT_BASE_DIR"])
    experience_list_path = base_dir / "prompts" / "empty_rewrites.json"

    # First iteration
    log_path = exp_base_dir / "log.txt"
    log_path.write_text("")
    with open(log_path, "a") as f:
        f.write(f"{init_exp_date}\n")

    run_first.run(
        exp_date=init_exp_date,
        exp_base_dir=exp_base_dir,
        experience_list_path=experience_list_path,
        profile_mode=profile_mode,
        breadth=breadth,
        num_samples=num_samples,
        exp_n=exp_n,
        project_name=project_name,
        org_name=org_name,
        logfire_enabled=logfire_enabled,
        log_file=log_file,
        machine_config_path=machine_config_path,
        machine_config_preset=machine_config_preset,
    )

    # Subsequent iterations
    last_exp_date = init_exp_date
    for _ in range(1, iters + 1):
        current_exp_date = f"{exp_date_prefix}-{datetime.now(LA).strftime('%m-%d-%H-%M')}"

        with open(log_path, "a") as f:
            f.write(f"{current_exp_date}\n")

        experience_list_path = exp_base_dir / last_exp_date / "rewrites" / "aggregated_rewrites_list.json"
        (exp_base_dir / current_exp_date).mkdir(parents=True, exist_ok=True)

        run_init.run(
            exp_date=current_exp_date,
            last_exp_date=last_exp_date,
            exp_base_dir=exp_base_dir,
            project_name=project_name,
            org_name=org_name,
            logfire_enabled=logfire_enabled,
            log_file=log_file,
        )

        run_accum_rewrites.run(
            exp_date=current_exp_date,
            exp_base_dir=exp_base_dir,
            profile_mode=profile_mode,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            topk=topk,
            topk_candidates=topk_candidates,
            project_name=project_name,
            rel_tol=rel_tol,
            org_name=org_name,
            logfire_enabled=logfire_enabled,
            log_file=log_file,
            machine_config_path=machine_config_path,
            machine_config_preset=machine_config_preset,
        )

        run_body.run(
            exp_date=current_exp_date,
            exp_base_dir=exp_base_dir,
            experience_list_path=experience_list_path,
            profile_mode=profile_mode,
            breadth=breadth,
            num_samples=num_samples,
            exp_n=exp_n,
            project_name=project_name,
            org_name=org_name,
            logfire_enabled=logfire_enabled,
            log_file=log_file,
            machine_config_path=machine_config_path,
            machine_config_preset=machine_config_preset,
        )

        last_exp_date = current_exp_date


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experiment loop")
    parser.add_argument("--exp_base_dir", type=Path, required=True)
    parser.add_argument("--init_exp_date", type=str, required=True)
    parser.add_argument("--exp_date_prefix", type=str, required=True)
    parser.add_argument("--profile_mode", type=str, required=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--rel_tol", type=float, required=True)
    parser.add_argument("--org_name", type=str, required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--breadth", type=int, required=True)
    parser.add_argument("--topk_candidates", type=int, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--max_threshold", type=float, required=True)
    parser.add_argument("--min_threshold", type=float, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--exp_n", type=int, required=True)
    parser.add_argument("--log_file", type=Path, required=True)
    parser.add_argument("--machine_config_path", type=str, default=None)
    parser.add_argument("--machine_config_preset", type=str, default="default")
    args = parser.parse_args()

    run(
        exp_base_dir=args.exp_base_dir,
        init_exp_date=args.init_exp_date,
        exp_date_prefix=args.exp_date_prefix,
        profile_mode=args.profile_mode,
        project_name=args.project_name,
        rel_tol=args.rel_tol,
        org_name=args.org_name,
        iters=args.iters,
        breadth=args.breadth,
        topk_candidates=args.topk_candidates,
        num_samples=args.num_samples,
        max_threshold=args.max_threshold,
        min_threshold=args.min_threshold,
        topk=args.topk,
        exp_n=args.exp_n,
        log_file=args.log_file,
        machine_config_path=args.machine_config_path,
        machine_config_preset=args.machine_config_preset,
    )
