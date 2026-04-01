"""Run the full experiment loop: first iteration + N subsequent iterations."""

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.loop import accum_rewrites, body_iter, first_iter, init_iter
from src.loop.logging_config import setup_problem_logger

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
    machine_config_path: str = None,
    machine_config_preset: str = "default",
    resume: bool = False,
    include_baseline: bool = False,
) -> None:
    setup_problem_logger(log_file)

    base_dir = Path(os.environ["ACCELOPT_BASE_DIR"])
    log_path = exp_base_dir / "log.txt"

    # Common kwargs passed to first_iter / body_iter / accum_rewrites
    common = dict(
        exp_base_dir=exp_base_dir,
        profile_mode=profile_mode,
        project_name=project_name,
        org_name=org_name,
        logfire_enabled=logfire_enabled,
        log_file=log_file,
        machine_config_path=machine_config_path,
        machine_config_preset=machine_config_preset,
        include_baseline=include_baseline,
    )

    if resume:
        assert log_path.exists(), f"Cannot resume: log.txt not found at {log_path}"
        entries = [e.strip() for e in log_path.read_text().splitlines() if e.strip()]
        assert entries, f"Cannot resume: log.txt is empty at {log_path}"
        remaining = iters - (len(entries) - 1)
        if remaining <= 0:
            print(f">>> Already complete ({len(entries)} iterations). Nothing to resume.")
            return
        print(f">>> RESUME: {len(entries)} iterations done, {remaining} remaining")
        last_exp_date = entries[-1]
    else:
        # First iteration
        experience_list_path = base_dir / "prompts" / "empty_rewrites.json"

        if not log_path.exists():
            log_path.write_text("")
        with open(log_path, "a") as f:
            f.write(f"{init_exp_date}\n")

        first_iter.run(
            exp_date=init_exp_date,
            experience_list_path=experience_list_path,
            breadth=breadth,
            num_samples=num_samples,
            exp_n=exp_n,
            rel_tol=rel_tol,
            **common,
        )

        last_exp_date = init_exp_date
        remaining = iters

    # Subsequent iterations
    for _ in range(remaining):
        current_exp_date = f"{exp_date_prefix}-{datetime.now(LA).strftime('%m-%d-%H-%M')}"

        with open(log_path, "a") as f:
            f.write(f"{current_exp_date}\n")

        experience_list_path = exp_base_dir / last_exp_date / "rewrites" / "aggregated_rewrites_list.json"
        (exp_base_dir / current_exp_date).mkdir(parents=True, exist_ok=True)

        init_iter.run(
            exp_date=current_exp_date,
            last_exp_date=last_exp_date,
            exp_base_dir=exp_base_dir,
            project_name=project_name,
            org_name=org_name,
            logfire_enabled=logfire_enabled,
            log_file=log_file,
        )

        accum_rewrites.run(
            exp_date=current_exp_date,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            topk=topk,
            topk_candidates=topk_candidates,
            rel_tol=rel_tol,
            **common,
        )

        body_iter.run(
            exp_date=current_exp_date,
            experience_list_path=experience_list_path,
            breadth=breadth,
            num_samples=num_samples,
            exp_n=exp_n,
            rel_tol=rel_tol,
            **common,
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
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in log.txt")
    parser.add_argument("--include_baseline", action="store_true", help="Include baseline kernel code in prompts")
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
        resume=args.resume,
        include_baseline=args.include_baseline,
    )
