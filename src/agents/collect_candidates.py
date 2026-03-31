# scripts/collect_candidates.py
import os
import argparse
import pandas as pd
import json
from src.utils import get_case_name

parser = argparse.ArgumentParser()
parser.add_argument("--output_candidates_path", type=str, required=True)
parser.add_argument("--output_profile_path", type=str, required=True)
parser.add_argument("--profile_mode", type=str, default="symbolic", choices=["symbolic", "cycle_accurate"])
parser.add_argument("--mode", type=str, default="construct")
parser.add_argument("--benchmarks", type=str, default="all",
                    help="Comma-separated benchmark names, or 'all' for every benchmark with a baseline")
parser.add_argument("--presets", type=str, default="all",
                    help="Comma-separated preset names, or 'all' for every preset per benchmark")
parser.add_argument("--machine_config_path", type=str, default=None, help="Path to machine_config.yaml")
parser.add_argument("--machine_config_preset", type=str, default="default", help="Preset name in machine_config.yaml")
parser.add_argument("--pipeline", type=str, default="pytorch-step")
parser.add_argument("--stage_config", type=str, default=None, help="JSON dict of pipeline overrides for multi-stage execution")
parser.add_argument("--bench_dir", type=str, default=None, help="Override bench_dir from pipeline (e.g. NKIBench)")
args = parser.parse_args()

from src.pipeline_registry import resolve_pipeline
pipeline = resolve_pipeline(args.pipeline)
if args.stage_config:
    pipeline = {**pipeline, **json.loads(args.stage_config)}
if args.bench_dir:
    pipeline["bench_dir"] = args.bench_dir
    # Override problem_key and baseline_key to match the target bench suite's naming convention.
    _BENCH_DIR_KEYS = {
        "NKIBench": {"problem_key": "task", "baseline_key": "kernel"},
        "StepBench": {"problem_key": "problem", "baseline_key": "baseline"},
    }
    assert args.bench_dir in _BENCH_DIR_KEYS, f"Unknown bench_dir: {args.bench_dir}"
    for key, val in _BENCH_DIR_KEYS[args.bench_dir].items():
        pipeline[key] = val
    # When profiling NKI on StepBench, use nki_baseline (not the STeP baseline)
    if args.bench_dir == "StepBench" and pipeline["profiler"] == "nki":
        pipeline["baseline_key"] = "nki_baseline"


def construct_table():
    requested_benchmarks = None if args.benchmarks == "all" else set(args.benchmarks.split(","))
    requested_presets = None if args.presets == "all" else set(args.presets.split(","))
    output_rows = []

    bench_dir = os.path.join(os.getenv("ACCELOPT_BASE_DIR"), pipeline["bench_dir"])
    if pipeline["bench_dir"] == "NKIBench":
        from NKIBench.loader import load_config
    else:
        from StepBench.loader import load_config
    config = load_config()
    kernel_key = pipeline["baseline_key"]
    task_key = pipeline["problem_key"]

    for bench_name, bench_info in config.items():
        kernel_path = bench_info.get(kernel_key)
        if not kernel_path:
            continue

        if requested_benchmarks is not None and bench_name not in requested_benchmarks:
            continue

        task_path = bench_info[task_key]

        # STeP baseline path (always under "baseline" key) — may differ from kernel_path
        # when the active profiler uses a different baseline (e.g., nki_baseline on StepBench).
        step_baseline_path = bench_info.get("baseline", "")
        if step_baseline_path and step_baseline_path != kernel_path:
            step_baseline_path = os.path.join(bench_dir, step_baseline_path)
        else:
            step_baseline_path = ""

        for preset_name, preset_dims in bench_info["presets"].items():
            if requested_presets is not None and preset_name not in requested_presets:
                continue
            case_name = get_case_name(bench_name, preset_dims)
            row = {
                "problem": bench_name,
                "preset": preset_name,
                "values": json.dumps(preset_dims),
                "task": os.path.join(bench_dir, task_path),
                "kernel": os.path.join(bench_dir, kernel_path),
                "step_baseline": step_baseline_path,
                "case_name": case_name,
                "service_name": case_name + "_ID0",
            }
            output_rows.append(row)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(args.output_candidates_path, index=False)


def construct_profile_table():
    df = pd.read_csv(args.output_candidates_path)
    output_rows = []

    if pipeline["profiler"] == "nki":
        from src.nki_kernel_wrapper import NKIKernel
        is_stepbench = pipeline.get("bench_dir") == "StepBench"
        for index, row in df.iterrows():
            dims = json.loads(row["values"])
            nki_kernel = NKIKernel(row["kernel"], row["task"])
            nki_kernel.rel_tol = 3e-5 if "mamba" in row["problem"] else 2e-5
            if is_stepbench:
                nki_kernel.profile_stepbench(dims, [])
            else:
                nki_kernel.profile([])
            props = nki_kernel.res
            profile_data = {"profile": json.dumps(props.metadata)}
            output_rows.append({**row, **profile_data})
    else:
        from src.step_kernel_wrapper import StepKernel, ProfileMode
        profile_mode = ProfileMode(args.profile_mode)
        for index, row in df.iterrows():
            dims = json.loads(row["values"])
            with open(row["kernel"], "r") as f:
                step_code = f.read()
            step_kernel = StepKernel(
                step_code=step_code,
                problem_path=row["task"],
                profile_mode=profile_mode,
                dims=dims,
                machine_config_path=args.machine_config_path,
                machine_config_preset=args.machine_config_preset,
            )
            props = step_kernel.profile()
            profile_data = {"profile": json.dumps(props.metadata)}
            output_rows.append({**row, **profile_data})

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(args.output_profile_path, index=False)


if __name__ == "__main__":
    if args.mode == "construct":
        construct_table()
        construct_profile_table()
    elif args.mode == "collect":
        construct_table()
    elif args.mode == "profile":
        construct_profile_table()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
