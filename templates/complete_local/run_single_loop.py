"""Run the full experiment loop: first iteration + N subsequent iterations.

For multi-stage pipelines (with a middleend), runs two sequential stages:
  Stage 1: Optimize using middleend prompts/profiler (e.g., STeP IR)
  Stage 2: Optimize using backend prompts/profiler (e.g., NKI), starting from
            Stage 1's best kernel as the new baseline.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# Ensure AccelOptStep root is importable when run standalone
_accelopt_root = Path(__file__).resolve().parent.parent.parent
if str(_accelopt_root) not in sys.path:
    sys.path.insert(0, str(_accelopt_root))

from templates.complete_local import run_accum_rewrites, run_body, run_first, run_init
from templates.complete_local.logging_config import setup_problem_logger

LA = ZoneInfo("America/Los_Angeles")


def _reprofile_baselines(profile_csv: Path, pipeline_cfg: dict, profile_mode: str,
                         machine_config_path: str | None, machine_config_preset: str) -> None:
    """Re-profile baselines in profile_results.csv with a different profiler.

    Used when a stage overrides the profiler (e.g., middleend stage needs STeP
    profiles but the CSV was generated with NKI profiles).
    """
    print(f">>> Re-profiling baselines with {pipeline_cfg['profiler']} profiler...")
    df = pd.read_csv(profile_csv)
    new_rows = []

    if pipeline_cfg["profiler"] == "nki":
        from accelopt.nki_kernel_wrapper import NKIKernel
        for _, row in df.iterrows():
            nki_kernel = NKIKernel(row["kernel"], row["task"])
            nki_kernel.rel_tol = 2e-5
            nki_kernel.profile([])
            new_rows.append({**row, "profile": json.dumps(nki_kernel.res.metadata)})
    else:
        from accelopt.step_kernel_wrapper import StepKernel, ProfileMode
        pm = ProfileMode.SYMBOLIC if profile_mode == "symbolic" else ProfileMode.CYCLE_ACCURATE
        for _, row in df.iterrows():
            dims = json.loads(row["values"])
            with open(row["kernel"]) as f:
                code = f.read()
            step_kernel = StepKernel(step_code=code, problem_path=row["task"],
                                     profile_mode=pm, dims=dims,
                                     machine_config_path=machine_config_path,
                                     machine_config_preset=machine_config_preset)
            props = step_kernel.profile()
            new_rows.append({**row, "profile": json.dumps(props.metadata)})

    pd.DataFrame(new_rows).to_csv(profile_csv, index=False)
    print(f">>> Done. {len(new_rows)} baselines re-profiled.")


def _run_stage(
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
    pipeline: str = "pytorch-step",
    stage_config: dict | None = None,
    _resume_last_exp_date: str | None = None,
    _resume_remaining_iters: int | None = None,
) -> str:
    """Run one stage of the experiment loop (first iteration + N subsequent).

    When _resume_last_exp_date is set, skips the first iteration and starts
    subsequent iterations from that checkpoint, running only
    _resume_remaining_iters iterations.

    Returns the exp_date of the last iteration (for extracting best results).
    """
    from pipeline_registry import resolve_pipeline
    pipeline_cfg = resolve_pipeline(pipeline)
    if stage_config:
        pipeline_cfg = {**pipeline_cfg, **stage_config}

    base_dir = Path(os.environ["ACCELOPT_BASE_DIR"])
    log_path = exp_base_dir / "log.txt"

    if _resume_last_exp_date is not None:
        # Resume path: skip first iteration, continue from last checkpoint
        assert _resume_remaining_iters is not None
        last_exp_date = _resume_last_exp_date
        remaining_iters = _resume_remaining_iters
        print(f">>> Resuming stage from {_resume_last_exp_date} ({remaining_iters} iterations remaining)")
    else:
        # Normal path: run first iteration
        experience_list_path = base_dir / "prompts" / "empty_rewrites.json"

        # Note: we do NOT re-profile baselines when stage_config overrides the profiler.
        # Baselines are in the backend's format (e.g., NKI kernels) and can't be profiled
        # with a different profiler (e.g., STeP). The original profile data is kept as-is
        # for LLM context. The executor profiles new candidates with the stage's profiler.

        if not log_path.exists():
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
            pipeline=pipeline,
            stage_config=stage_config,
        )

        last_exp_date = init_exp_date
        remaining_iters = iters

    # Subsequent iterations
    for _ in range(remaining_iters):
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
            pipeline=pipeline,
            stage_config=stage_config,
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
            pipeline=pipeline,
            stage_config=stage_config,
        )

        last_exp_date = current_exp_date

    return last_exp_date


def _extract_best_kernel_path(exp_base_dir: Path, last_exp_date: str, speedup_metric: str) -> Path | None:
    """Find the best-performing kernel from the final iteration's candidates.

    Returns the path to the best kernel file, or None if no valid candidates.
    """
    candidates_dir = exp_base_dir / last_exp_date / "candidates"
    profile_csv = candidates_dir / "profile_results.csv"
    if not profile_csv.exists():
        return None

    df = pd.read_csv(profile_csv)
    if df.empty:
        return None

    # Parse profile JSON and find best by speedup metric
    best_path = None
    best_score = None
    for _, row in df.iterrows():
        profile = json.loads(row["profile"])
        score = profile.get(speedup_metric)
        if score is None:
            continue
        # For latency, lower is better; for cycles, lower is better
        if best_score is None or score < best_score:
            best_score = score
            best_path = row.get("kernel")

    if best_path and Path(best_path).exists():
        return Path(best_path)
    return None


def _setup_stage2_baseline(
    exp_base_dir: Path,
    stage1_best_kernel: Path,
    stage2_exp_date: str,
    original_profile_csv: Path,
    pipeline_cfg: dict,
) -> None:
    """Set up Stage 2's initial candidates directory with the Stage 1 best kernel as baseline.

    Copies the Stage 1 best kernel to the Stage 2 experiment dir and creates
    a profile_results.csv with it as the baseline (profiled with the backend profiler).
    """
    stage2_candidates_dir = exp_base_dir / stage2_exp_date / "candidates"
    stage2_candidates_dir.mkdir(parents=True, exist_ok=True)

    # Read original profile CSV to get problem metadata (task paths, values, etc.)
    original_df = pd.read_csv(original_profile_csv)
    assert len(original_df) == 1, f"Expected single problem in profile CSV, got {len(original_df)}"
    row = original_df.iloc[0].to_dict()

    # Copy stage1 best kernel to a stable location
    stage2_kernel_dir = exp_base_dir / stage2_exp_date / "stage1_best"
    stage2_kernel_dir.mkdir(parents=True, exist_ok=True)
    dest_kernel = stage2_kernel_dir / stage1_best_kernel.name
    shutil.copy2(stage1_best_kernel, dest_kernel)

    # Profile the stage1 kernel with the backend profiler
    if pipeline_cfg["profiler"] == "nki":
        from accelopt.nki_kernel_wrapper import NKIKernel
        nki_kernel = NKIKernel(str(dest_kernel), row["task"])
        nki_kernel.rel_tol = 2e-5
        nki_kernel.profile([])
        profile_data = json.dumps(nki_kernel.res.metadata)
    else:
        from accelopt.step_kernel_wrapper import StepKernel, ProfileMode
        dims = json.loads(row["values"])
        with open(str(dest_kernel)) as f:
            code = f.read()
        step_kernel = StepKernel(step_code=code, problem_path=row["task"],
                                 profile_mode=ProfileMode.CYCLE_ACCURATE, dims=dims)
        props = step_kernel.profile()
        profile_data = json.dumps(props.metadata)

    # Write profile CSV with stage1 kernel as baseline
    new_row = {**row, "kernel": str(dest_kernel), "profile": profile_data}
    pd.DataFrame([new_row]).to_csv(stage2_candidates_dir / "profile_results.csv", index=False)


def _parse_log_entries(log_path: Path, exp_date_prefix: str) -> dict:
    """Parse log.txt and classify entries by stage.

    Returns dict with:
        entries: list of all entries
        s1_entries: entries for stage 1 (first iteration + s1-suffix)
        s2_entries: entries for stage 2 (s2-suffix)
    """
    entries = [e.strip() for e in log_path.read_text().splitlines() if e.strip()]
    s2_entries = [e for e in entries if f"{exp_date_prefix}-s2-" in e]
    s1_entries = [e for e in entries if e not in s2_entries]
    return {"entries": entries, "s1_entries": s1_entries, "s2_entries": s2_entries}


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
    pipeline: str = "pytorch-step",
    middleend_iters: int | None = None,
    resume: bool = False,
) -> None:
    setup_problem_logger(log_file)
    from pipeline_registry import resolve_pipeline, get_stage_configs
    pipeline_cfg = resolve_pipeline(pipeline)

    stage1_config, stage2_config = get_stage_configs(pipeline)
    is_multi_stage = stage1_config is not None

    # Common kwargs for _run_stage calls
    common_kwargs = dict(
        exp_base_dir=exp_base_dir,
        profile_mode=profile_mode,
        project_name=project_name,
        rel_tol=rel_tol,
        org_name=org_name,
        logfire_enabled=logfire_enabled,
        breadth=breadth,
        topk_candidates=topk_candidates,
        num_samples=num_samples,
        max_threshold=max_threshold,
        min_threshold=min_threshold,
        topk=topk,
        exp_n=exp_n,
        log_file=log_file,
        machine_config_preset=machine_config_preset,
        pipeline=pipeline,
    )

    if resume:
        log_path = exp_base_dir / "log.txt"
        assert log_path.exists(), f"Cannot resume: log.txt not found at {log_path}"
        parsed = _parse_log_entries(log_path, exp_date_prefix)
        assert parsed["entries"], f"Cannot resume: log.txt is empty at {log_path}"
        return _run_resume(
            parsed=parsed,
            init_exp_date=init_exp_date,
            exp_date_prefix=exp_date_prefix,
            iters=iters,
            middleend_iters=middleend_iters,
            is_multi_stage=is_multi_stage,
            pipeline_cfg=pipeline_cfg,
            stage1_config=stage1_config,
            stage2_config=stage2_config,
            machine_config_path=machine_config_path,
            common_kwargs=common_kwargs,
        )

    if not is_multi_stage:
        # Single-stage: run the loop directly (original behavior)
        _run_stage(
            init_exp_date=init_exp_date,
            exp_date_prefix=exp_date_prefix,
            iters=iters,
            machine_config_path=machine_config_path,
            **common_kwargs,
        )
        return

    _run_multi_stage_fresh(
        init_exp_date=init_exp_date,
        exp_date_prefix=exp_date_prefix,
        iters=iters,
        middleend_iters=middleend_iters,
        pipeline_cfg=pipeline_cfg,
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        machine_config_path=machine_config_path,
        common_kwargs=common_kwargs,
    )


def _run_multi_stage_fresh(
    init_exp_date: str,
    exp_date_prefix: str,
    iters: int,
    middleend_iters: int | None,
    pipeline_cfg: dict,
    stage1_config: dict,
    stage2_config: dict,
    machine_config_path: str | None,
    common_kwargs: dict,
) -> None:
    """Run a fresh multi-stage pipeline (Stage 1 middleend, then Stage 2 backend)."""
    exp_base_dir = common_kwargs["exp_base_dir"]

    print(f"\n{'='*60}")
    print(f"MULTI-STAGE PIPELINE: {pipeline_cfg['pipeline']}")
    print(f"  Stage 1 (middleend={pipeline_cfg['middleend']}): "
          f"profiler={stage1_config['profiler']}, prompts={stage1_config['prompts_subdir']}")
    print(f"  Stage 2 (backend={pipeline_cfg['backend']}): "
          f"profiler={pipeline_cfg['profiler']}, prompts={pipeline_cfg['prompts_subdir']}")
    print(f"{'='*60}\n")

    # For Stage 1 machine config, check if middleend needs it
    stage1_machine_config_path = machine_config_path
    if stage1_config.get("needs_machine_config") and not machine_config_path:
        base_dir = Path(os.environ["ACCELOPT_BASE_DIR"])
        stage1_machine_config_path = str(base_dir / "StepBench" / "machine_config.yaml")

    # ---- Stage 1: Middleend ----
    print(f">>> STAGE 1: {pipeline_cfg['middleend'].upper()} middleend optimization")
    stage1_last_date = _run_stage(
        init_exp_date=init_exp_date,
        exp_date_prefix=f"{exp_date_prefix}-s1",
        iters=middleend_iters if middleend_iters is not None else iters,
        machine_config_path=stage1_machine_config_path,
        stage_config=stage1_config,
        **common_kwargs,
    )

    # Transition Stage 1 → Stage 2
    stage2_init_date = _setup_stage2_from_stage1(
        exp_base_dir=exp_base_dir,
        init_exp_date=init_exp_date,
        exp_date_prefix=exp_date_prefix,
        stage1_last_date=stage1_last_date,
        stage1_config=stage1_config,
        pipeline_cfg=pipeline_cfg,
    )

    # ---- Stage 2: Backend ----
    print(f"\n>>> STAGE 2: {pipeline_cfg['backend'].upper()} backend optimization")
    _run_stage(
        init_exp_date=stage2_init_date,
        exp_date_prefix=f"{exp_date_prefix}-s2",
        iters=iters,
        machine_config_path=machine_config_path,
        stage_config=stage2_config if stage2_config else None,
        **common_kwargs,
    )


def _setup_stage2_from_stage1(
    exp_base_dir: Path,
    init_exp_date: str,
    exp_date_prefix: str,
    stage1_last_date: str,
    stage1_config: dict,
    pipeline_cfg: dict,
) -> str:
    """Extract Stage 1 best kernel and set up Stage 2 initial directory.

    Returns the stage2_init_date string.
    """
    stage1_best = _extract_best_kernel_path(
        exp_base_dir, stage1_last_date,
        speedup_metric=stage1_config["speedup_metric"],
    )

    if stage1_best is None:
        print(">>> STAGE 1 produced no valid kernels. Falling back to original baseline for Stage 2.")

    stage2_init_date = f"{exp_date_prefix}-s2-{datetime.now(LA).strftime('%m-%d-%H-%M')}"
    (exp_base_dir / stage2_init_date).mkdir(parents=True, exist_ok=True)

    if stage1_best is not None:
        original_profile_csv = exp_base_dir / init_exp_date / "candidates" / "profile_results.csv"
        _setup_stage2_baseline(
            exp_base_dir=exp_base_dir,
            stage1_best_kernel=stage1_best,
            stage2_exp_date=stage2_init_date,
            original_profile_csv=original_profile_csv,
            pipeline_cfg=pipeline_cfg,
        )
        print(f"    Stage 2 baseline: {stage1_best}")
    else:
        original_candidates = exp_base_dir / init_exp_date / "candidates"
        stage2_candidates = exp_base_dir / stage2_init_date / "candidates"
        stage2_candidates.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            original_candidates / "profile_results.csv",
            stage2_candidates / "profile_results.csv",
        )

    return stage2_init_date


def _run_resume(
    parsed: dict,
    init_exp_date: str,
    exp_date_prefix: str,
    iters: int,
    middleend_iters: int | None,
    is_multi_stage: bool,
    pipeline_cfg: dict,
    stage1_config: dict | None,
    stage2_config: dict,
    machine_config_path: str | None,
    common_kwargs: dict,
) -> None:
    """Resume an interrupted experiment from checkpoint state in log.txt."""
    exp_base_dir = common_kwargs["exp_base_dir"]

    if not is_multi_stage:
        # Single-stage resume
        entries = parsed["entries"]
        remaining = iters - (len(entries) - 1)
        if remaining <= 0:
            print(f">>> Already complete ({len(entries)} iterations). Nothing to resume.")
            return
        print(f">>> RESUME: single-stage, {len(entries)} iterations done, {remaining} remaining")
        _run_stage(
            init_exp_date=init_exp_date,
            exp_date_prefix=exp_date_prefix,
            iters=iters,
            machine_config_path=machine_config_path,
            _resume_last_exp_date=entries[-1],
            _resume_remaining_iters=remaining,
            **common_kwargs,
        )
        return

    # Multi-stage resume
    s1_entries = parsed["s1_entries"]
    s2_entries = parsed["s2_entries"]
    stage1_iters = middleend_iters if middleend_iters is not None else iters
    stage1_expected = 1 + stage1_iters  # first iteration + subsequent

    # For Stage 1 machine config
    stage1_machine_config_path = machine_config_path
    if stage1_config.get("needs_machine_config") and not machine_config_path:
        base_dir = Path(os.environ["ACCELOPT_BASE_DIR"])
        stage1_machine_config_path = str(base_dir / "StepBench" / "machine_config.yaml")

    stage1_complete = len(s1_entries) >= stage1_expected

    if not stage1_complete:
        # Resume Stage 1
        s1_remaining = stage1_iters - (len(s1_entries) - 1)
        print(f">>> RESUME: Stage 1 incomplete ({len(s1_entries)}/{stage1_expected}), {s1_remaining} iterations remaining")
        stage1_last_date = _run_stage(
            init_exp_date=init_exp_date,
            exp_date_prefix=f"{exp_date_prefix}-s1",
            iters=stage1_iters,
            machine_config_path=stage1_machine_config_path,
            stage_config=stage1_config,
            _resume_last_exp_date=s1_entries[-1],
            _resume_remaining_iters=s1_remaining,
            **common_kwargs,
        )

        # Stage 1 now complete — set up and run full Stage 2
        stage2_init_date = _setup_stage2_from_stage1(
            exp_base_dir=exp_base_dir,
            init_exp_date=init_exp_date,
            exp_date_prefix=exp_date_prefix,
            stage1_last_date=stage1_last_date,
            stage1_config=stage1_config,
            pipeline_cfg=pipeline_cfg,
        )
        print(f"\n>>> STAGE 2: {pipeline_cfg['backend'].upper()} backend optimization")
        _run_stage(
            init_exp_date=stage2_init_date,
            exp_date_prefix=f"{exp_date_prefix}-s2",
            iters=iters,
            machine_config_path=machine_config_path,
            stage_config=stage2_config if stage2_config else None,
            **common_kwargs,
        )
        return

    # Stage 1 complete — check Stage 2
    if not s2_entries:
        # Stage 1 done but Stage 2 never started
        print(f">>> RESUME: Stage 1 complete, Stage 2 not started. Setting up Stage 2...")
        stage1_last_date = s1_entries[-1]
        stage2_init_date = _setup_stage2_from_stage1(
            exp_base_dir=exp_base_dir,
            init_exp_date=init_exp_date,
            exp_date_prefix=exp_date_prefix,
            stage1_last_date=stage1_last_date,
            stage1_config=stage1_config,
            pipeline_cfg=pipeline_cfg,
        )
        print(f"\n>>> STAGE 2: {pipeline_cfg['backend'].upper()} backend optimization")
        _run_stage(
            init_exp_date=stage2_init_date,
            exp_date_prefix=f"{exp_date_prefix}-s2",
            iters=iters,
            machine_config_path=machine_config_path,
            stage_config=stage2_config if stage2_config else None,
            **common_kwargs,
        )
        return

    # Stage 2 partially done
    s2_remaining = iters - (len(s2_entries) - 1)
    if s2_remaining <= 0:
        print(f">>> Already complete (Stage 1: {len(s1_entries)}, Stage 2: {len(s2_entries)} iterations). Nothing to resume.")
        return

    print(f">>> RESUME: Stage 1 complete ({len(s1_entries)}), Stage 2 incomplete ({len(s2_entries)}/{1 + iters}), {s2_remaining} iterations remaining")
    _run_stage(
        init_exp_date=s2_entries[0],
        exp_date_prefix=f"{exp_date_prefix}-s2",
        iters=iters,
        machine_config_path=machine_config_path,
        stage_config=stage2_config if stage2_config else None,
        _resume_last_exp_date=s2_entries[-1],
        _resume_remaining_iters=s2_remaining,
        **common_kwargs,
    )


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
    parser.add_argument("--pipeline", type=str, default="pytorch-step")
    parser.add_argument("--middleend_iters", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in log.txt")
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
        pipeline=args.pipeline,
        middleend_iters=args.middleend_iters,
        resume=args.resume,
    )
