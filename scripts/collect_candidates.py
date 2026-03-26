import os
import argparse
import pandas as pd
import json
from accelopt.utils import get_case_name
from accelopt.step_kernel_wrapper import StepKernel, ProfileMode

parser = argparse.ArgumentParser()
parser.add_argument("--summary_path", type=str, default="")
parser.add_argument("--output_candidates_path", type=str, required=True)
parser.add_argument("--output_profile_path", type=str, required=True)
parser.add_argument("--profile_mode", type=str, default="symbolic", choices=["symbolic", "cycle_accurate"])
parser.add_argument("--mode", type=str, default="construct")
args = parser.parse_args()

stepbench_base_path = os.path.join(os.getenv("ACCELOPT_BASE_DIR"), "StepBench")


output_dict = []
problems = [
    "gemm_square", "gemm_rectangular", "gemm_batched", "gemm_large_k",
    "gemm_small_k", "gemm_3d", "relu", "sigmoid", "swish", "gelu",
    "layernorm", "rmsnorm", "softmax", "sdpa", "flash_attention",
    "gemm_silu", "gemm_gelu_softmax", "gemm_scale_residual",
    "gemm_rmsnorm", "gemm_swish_groupnorm", "moe_gemm", "moe_full",
    "mamba_ssm", "transformer_ffn", "transformer_layer",
]

def construct_table():
    base_summary_path = os.path.join(stepbench_base_path, "summary.json")
    with open(base_summary_path, "r") as f:
        summary = json.load(f)

    output_rows = []
    for problem_name, problem_info in summary.items():
        for case_id, case_info in problem_info["cases"].items():
            for single_impl in case_info["impls"]:
                if problem_name not in problems:
                    continue
                case_name = get_case_name(problem_name, case_info["values"])
                row = {
                    "problem": problem_name,
                    "values": json.dumps(case_info["values"]),
                    "case_id": case_id,
                    "task": os.path.join(stepbench_base_path, single_impl["task"]),
                    "kernel": os.path.join(stepbench_base_path, single_impl["kernel"]),
                    "case_name": case_name,
                    "service_name": case_name + "_ID0"
                }
                output_rows.append(row)
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(args.output_candidates_path, index=False)


def construct_profile_table():
    profile_mode = ProfileMode(args.profile_mode)
    df = pd.read_csv(args.output_candidates_path)
    output_rows = []
    for index, row in df.iterrows():
        with open(row["kernel"], "r") as f:
            step_code = f.read()
        step_kernel = StepKernel(
            step_code=step_code,
            problem_path=row["task"],
            profile_mode=profile_mode,
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
