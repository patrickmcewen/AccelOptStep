import pandas as pd
import os
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse

LA = ZoneInfo("America/Los_Angeles")

parser = argparse.ArgumentParser()
parser.add_argument("--project_name", type=str, required=True)
parser.add_argument("--org_name", type=str, required=True)
parser.add_argument("--exp_date_base", type=str, required=True)
parser.add_argument("--profile_mode", type=str, default="symbolic",
                    choices=["symbolic", "cycle_accurate"])
args = parser.parse_args()

project_name = args.project_name
org_name = args.org_name
exp_date_base = args.exp_date_base
PROFILE_MODE = args.profile_mode

ITERS = 15
BREADTH = 12
TOPK_CANDIDATES = 6
NUM_SAMPLES = 2
MAX_THRESHOLD = 1.04
MIN_THRESHOLD = 1.15
TOPK = 8
EXP_N = 16
REL_TOL = 2e-5

config_dirs = [
    "./configs",
]

proxy_problem_list_file = "candidates.csv"
proxy_profile_results_file = "profile_results.csv"
proxy_problem_list_df = pd.read_csv(proxy_problem_list_file)
proxy_profile_results_df = pd.read_csv(proxy_profile_results_file)

exp_base_dir = f"../checkpoints/{exp_date_base}"
exp_base_dir = os.path.abspath(exp_base_dir)
ACCELOPT_BASE_DIR = os.getenv("ACCELOPT_BASE_DIR")
assert ACCELOPT_BASE_DIR is not None, "ACCELOPT_BASE_DIR environment variable must be set"
single_loop_exec = os.path.join(ACCELOPT_BASE_DIR, "templates", "complete_local", "run_single_loop.sh")

# Template substitution mapping: $N -> value
# Keys are sorted by descending N so that $15 is replaced before $1, etc.
SUBSTITUTIONS = {
    15: lambda _i, _row: str(EXP_N),
    14: lambda _i, _row: str(TOPK),
    13: lambda _i, _row: str(MIN_THRESHOLD),
    12: lambda _i, _row: str(MAX_THRESHOLD),
    11: lambda _i, _row: str(NUM_SAMPLES),
    10: lambda _i, _row: str(TOPK_CANDIDATES),
    9:  lambda _i, _row: str(BREADTH),
    8:  lambda _i, _row: str(ITERS),
    7:  lambda _i, _row: f'"{org_name}"',
    6:  lambda _i, _row: str(REL_TOL),
    5:  lambda _i, _row: f'"{project_name}"',
    4:  lambda _i, _row: PROFILE_MODE,
    3:  lambda _i, _row: f'"{_row["eval_prefix"]}"',
    2:  lambda _i, _row: f'"{_row["eval_first_exp_date"]}"',
    1:  lambda _i, _row: f'"{_row["new_exp_base_dir"]}"',
}

first_exp_date = datetime.now(LA).strftime("%m-%d-%H-%M")

for index, row in proxy_problem_list_df.iterrows():
    service_name = row["service_name"]

    new_exp_base_dir = os.path.join(exp_base_dir, service_name)
    os.makedirs(new_exp_base_dir, exist_ok=False)
    new_exp_config_dir = os.path.join(new_exp_base_dir, "configs")
    shutil.copytree(config_dirs[index % len(config_dirs)], new_exp_config_dir, dirs_exist_ok=True)
    eval_prefix = f"eval-{index}-{exp_date_base}"
    eval_first_exp_date = f"{eval_prefix}-{first_exp_date}"
    new_exp_candidates_dir = os.path.join(new_exp_base_dir, eval_first_exp_date, "candidates")
    os.makedirs(new_exp_candidates_dir, exist_ok=False)

    profile_results_df = proxy_profile_results_df[proxy_profile_results_df["service_name"] == service_name]
    profile_results_df.to_csv(os.path.join(new_exp_candidates_dir, "profile_results.csv"), index=False)

    cur_single_loop_exec_path = os.path.join(exp_base_dir, f"run_single_loop_{service_name}.sh")

    # Build a row-context dict for the substitution lambdas
    row_ctx = {
        "new_exp_base_dir": new_exp_base_dir,
        "eval_first_exp_date": eval_first_exp_date,
        "eval_prefix": eval_prefix,
    }

    with open(single_loop_exec, "r") as f:
        content = f.read()

    # Replace $N placeholders in descending order to avoid $1 matching inside $10, $11, etc.
    for n in sorted(SUBSTITUTIONS.keys(), reverse=True):
        content = content.replace(f"${n}", SUBSTITUTIONS[n](index, row_ctx))

    with open(cur_single_loop_exec_path, "w") as f:
        f.write(content)
