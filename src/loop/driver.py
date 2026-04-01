"""Run the full experiment loop: correctness phase + optimization phase."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.loop import accum_rewrites, body_iter, correctness_iter, init_iter
from src.loop.logging_config import setup_problem_logger
from src.loop.phase_state import (
    load_phase_state, save_phase_state, get_phase, init_service_state,
    record_correct_kernel, check_transition, get_best_correct_kernel,
)
from src.loop.scan_results import scan_for_correct_kernels

LA = ZoneInfo("America/Los_Angeles")


def _get_service_name(exp_base_dir: Path) -> str:
    """Extract service name from the exp_base_dir path (it's the directory name)."""
    return exp_base_dir.name


def _write_correct_baseline(exp_dir: Path, kernel_code: str, metadata: dict) -> None:
    """Write the correct kernel as the new baseline for optimization."""
    candidates_dir = exp_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / "correct_baseline.py").write_text(kernel_code)
    (candidates_dir / "correct_baseline_metadata.json").write_text(json.dumps(metadata))


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
    # Correctness-first phase parameters
    correctness_threshold: int = 1,
    correctness_max_fixup_attempts: int = 5,
    correctness_breadth: int | None = None,
    correctness_num_samples: int | None = None,
) -> None:
    setup_problem_logger(log_file)

    base_dir = Path(os.environ["ACCELOPT_BASE_DIR"])
    log_path = exp_base_dir / "log.txt"

    service_name = _get_service_name(exp_base_dir)

    # Load or initialize phase state
    phase_state = load_phase_state(exp_base_dir)
    phase_state = init_service_state(phase_state, service_name)

    # Resolve correctness-phase overrides
    c_breadth = correctness_breadth if correctness_breadth is not None else breadth
    c_num_samples = correctness_num_samples if correctness_num_samples is not None else num_samples

    # Common kwargs passed to body_iter / accum_rewrites
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
        # First iteration (always correctness phase)
        experience_list_path = base_dir / "prompts" / "empty_rewrites.json"

        if not log_path.exists():
            log_path.write_text("")
        with open(log_path, "a") as f:
            f.write(f"{init_exp_date}\n")

        correctness_iter.run(
            exp_date=init_exp_date,
            experience_list_path=experience_list_path,
            breadth=c_breadth,
            num_samples=c_num_samples,
            exp_n=exp_n,
            rel_tol=rel_tol,
            max_fixup_attempts=correctness_max_fixup_attempts,
            **common,
        )

        # Scan for correct kernels
        executor_results_path = exp_base_dir / init_exp_date / "executor_results.json"
        if executor_results_path.exists():
            correct = scan_for_correct_kernels(executor_results_path, init_exp_date)
            for k in correct:
                phase_state = record_correct_kernel(phase_state, service_name, k)
            check_transition(phase_state, service_name, correctness_threshold)
            save_phase_state(exp_base_dir, phase_state)

        last_exp_date = init_exp_date
        remaining = iters

    # Subsequent iterations
    for _ in range(remaining):
        current_exp_date = f"{exp_date_prefix}-{datetime.now(LA).strftime('%m-%d-%H-%M')}"

        with open(log_path, "a") as f:
            f.write(f"{current_exp_date}\n")

        phase = get_phase(phase_state, service_name)

        if phase == "correctness":
            # Correctness iteration — no accum_rewrites, use correctness prompts
            experience_list_path = exp_base_dir / last_exp_date / "rewrites" / "aggregated_rewrites_list.json"
            (exp_base_dir / current_exp_date).mkdir(parents=True, exist_ok=True)

            # Copy prior candidates (profile_results.csv) for planner context
            prior_candidates = exp_base_dir / last_exp_date / "candidates"
            current_candidates = exp_base_dir / current_exp_date / "candidates"
            current_candidates.mkdir(parents=True, exist_ok=True)
            prior_profile = prior_candidates / "profile_results.csv"
            if prior_profile.exists():
                shutil.copy2(prior_profile, current_candidates / "profile_results.csv")

            correctness_iter.run(
                exp_date=current_exp_date,
                experience_list_path=experience_list_path,
                breadth=c_breadth,
                num_samples=c_num_samples,
                exp_n=exp_n,
                rel_tol=rel_tol,
                max_fixup_attempts=correctness_max_fixup_attempts,
                **common,
            )

            # Scan for correct kernels
            executor_results_path = exp_base_dir / current_exp_date / "executor_results.json"
            if executor_results_path.exists():
                correct = scan_for_correct_kernels(executor_results_path, current_exp_date)
                for k in correct:
                    phase_state = record_correct_kernel(phase_state, service_name, k)
                transitioned = check_transition(phase_state, service_name, correctness_threshold)
                if transitioned:
                    # Write the best correct kernel as the new baseline
                    best = get_best_correct_kernel(phase_state, service_name)
                    assert best is not None, "Transitioned but no correct kernel found"
                    _write_correct_baseline(
                        exp_base_dir / current_exp_date,
                        best["code"],
                        best["metadata"],
                    )
                    print(f">>> Phase transition: correctness -> optimization (found {len(correct)} correct kernels)")
                save_phase_state(exp_base_dir, phase_state)

        else:
            # Optimization iteration — standard flow
            (exp_base_dir / current_exp_date).mkdir(parents=True, exist_ok=True)
            experience_list_path = exp_base_dir / last_exp_date / "rewrites" / "aggregated_rewrites_list.json"

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
    import argparse

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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include_baseline", action="store_true")
    parser.add_argument("--correctness_threshold", type=int, default=1)
    parser.add_argument("--correctness_max_fixup_attempts", type=int, default=5)
    parser.add_argument("--correctness_breadth", type=int, default=None)
    parser.add_argument("--correctness_num_samples", type=int, default=None)
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
        correctness_threshold=args.correctness_threshold,
        correctness_max_fixup_attempts=args.correctness_max_fixup_attempts,
        correctness_breadth=args.correctness_breadth,
        correctness_num_samples=args.correctness_num_samples,
    )
