# Collect and profile StepBench baselines for the optimization loop.
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
parser.add_argument("--machine_config_path", type=str, default=None)
parser.add_argument("--machine_config_preset", type=str, default="default")
parser.add_argument("--bench_dir", type=str, default=None, help="Override bench_dir (default: StepBench)")
args = parser.parse_args()

bench_dir_name = args.bench_dir or "StepBench"


def construct_table():
    requested_benchmarks = None if args.benchmarks == "all" else set(args.benchmarks.split(","))
    requested_presets = None if args.presets == "all" else set(args.presets.split(","))
    output_rows = []

    bench_dir = os.path.join(os.getenv("ACCELOPT_BASE_DIR"), bench_dir_name)
    from StepBench.loader import load_config
    config = load_config()

    for bench_name, bench_info in config.items():
        kernel_path = bench_info.get("baseline")
        if not kernel_path:
            continue

        if requested_benchmarks is not None and bench_name not in requested_benchmarks:
            continue

        task_path = bench_info["problem"]

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
                "case_name": case_name,
                "service_name": case_name + "_ID0",
            }
            output_rows.append(row)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(args.output_candidates_path, index=False)


def construct_profile_table():
    df = pd.read_csv(args.output_candidates_path)
    output_rows = []

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
        assert False, f"Invalid mode: {args.mode}"
