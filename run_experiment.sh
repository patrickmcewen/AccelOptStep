#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# AccelOptStep — Run the agentic optimization loop
# =============================================================================
#
# Usage:
#   bash run_experiment.sh [OPTIONS]
#
# This script:
#   1. Sets up environment (ACCELOPT_BASE_DIR, PYTHONPATH)
#   2. Generates profile_results.csv from baselines (if needed)
#   3. Scaffolds experiment directories via create_folders.py
#   4. Launches the beam search loop for each problem
#
# Prerequisites:
#   - vLLM server running at http://localhost:31001/v1 (or update configs/)
#   - STeP runtime: step_artifact/src on PYTHONPATH (auto-configured below)
#   - For cycle_accurate mode: Docker container with step_artifact
#
# =============================================================================

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACCELOPT_BASE_DIR="$SCRIPT_DIR"
STEP_ARTIFACT_SRC="${STEP_ARTIFACT_SRC:-$(realpath "$SCRIPT_DIR/../step_artifact/src")}"
export PYTHONPATH="${STEP_ARTIFACT_SRC}:${STEP_ARTIFACT_SRC}/step_py:${STEP_ARTIFACT_SRC}/sim:${STEP_ARTIFACT_SRC}/proto:${PYTHONPATH:-}"

# ── Configuration (override via environment or CLI) ──────────────────────────

# Logfire project/org (for tracing — set to dummy values if not using logfire)
PROJECT_NAME="${PROJECT_NAME:-acceloptstep}"
ORG_NAME="${ORG_NAME:-default}"

# Experiment tag (used for directory naming)
EXP_DATE_BASE="${EXP_DATE_BASE:-$(date +%Y-%m-%d-%H%M%S)}"

# Profiling backend: always runs both symbolic + cycle_accurate internally.
# This flag is kept for backward compatibility but is effectively ignored.
PROFILE_MODE="${PROFILE_MODE:-cycle_accurate}"

# ── Beam Search Parameters ───────────────────────────────────────────────────

# ITERS: Number of optimization iterations (after the first).
#   Each iteration: accumulate rewrites → select candidates → plan → execute.
#   More iterations = more optimization rounds. Typical: 5-15.
ITERS="${ITERS:-15}"

# BREADTH: Number of parallel optimization plans generated per problem per iteration.
#   Higher = more diverse plans but more LLM calls. Typical: 4-12.
BREADTH="${BREADTH:-12}"

# NUM_SAMPLES: Number of code implementations generated per plan.
#   Each plan is implemented NUM_SAMPLES times. Typical: 2-4.
NUM_SAMPLES="${NUM_SAMPLES:-2}"

# ── Selection Parameters ─────────────────────────────────────────────────────

# TOPK_CANDIDATES: Max candidates carried forward to the next iteration.
#   Controls beam width. Higher = more diversity. Typical: 4-8.
TOPK_CANDIDATES="${TOPK_CANDIDATES:-6}"

# TOPK: Number of optimization experiences to summarize and feed back into prompts.
#   Controls how many past successes/failures inform future plans. Typical: 4-8.
TOPK="${TOPK:-8}"

# EXP_N: Number of experience items sampled from the experience list for prompts.
#   Higher = more context for the LLM but longer prompts. Typical: 8-16.
EXP_N="${EXP_N:-16}"

# MAX_THRESHOLD: Minimum speedup to count as a "positive" rewrite (e.g., 1.04 = 4% faster).
MAX_THRESHOLD="${MAX_THRESHOLD:-1.04}"

# MIN_THRESHOLD: Speedups above this are "strong positives" for selection (e.g., 1.15 = 15% faster).
MIN_THRESHOLD="${MIN_THRESHOLD:-1.15}"

# ── Correctness tolerance ────────────────────────────────────────────────────
# REL_TOL: Relative tolerance for numerical correctness checking.
REL_TOL="${REL_TOL:-2e-5}"

# ── Dry run mode ─────────────────────────────────────────────────────────────
DRY_RUN="${DRY_RUN:-false}"

# =============================================================================
# Step 1: Generate profile_results.csv if it's empty (header-only)
# =============================================================================
EXPERIMENTS_DIR="$SCRIPT_DIR/experiments/full_complete_local"
PROFILE_CSV="$EXPERIMENTS_DIR/profile_results.csv"

line_count=$(wc -l < "$PROFILE_CSV" 2>/dev/null || echo "0")
if [ "$line_count" -le 1 ]; then
    echo ">>> Generating profile_results.csv (profiling baselines with mode=$PROFILE_MODE)..."
    python "$SCRIPT_DIR/scripts/collect_candidates.py" \
        --output_candidates_path "$EXPERIMENTS_DIR/candidates.csv" \
        --output_profile_path "$PROFILE_CSV" \
        --profile_mode "$PROFILE_MODE" \
        --mode construct
    echo ">>> Done. $(( $(wc -l < "$PROFILE_CSV") - 1 )) baselines profiled."
else
    echo ">>> profile_results.csv already populated ($(( line_count - 1 )) rows). Skipping profiling."
fi

# =============================================================================
# Step 2: Scaffold experiment directories
# =============================================================================
echo ""
echo ">>> Scaffolding experiment directories..."
echo "    EXP_DATE_BASE=$EXP_DATE_BASE"
echo "    PROFILE_MODE=$PROFILE_MODE"
echo "    ITERS=$ITERS  BREADTH=$BREADTH  NUM_SAMPLES=$NUM_SAMPLES"
echo "    TOPK_CANDIDATES=$TOPK_CANDIDATES  TOPK=$TOPK  EXP_N=$EXP_N"
echo ""

# Override create_folders.py constants via a temp wrapper
# (create_folders.py has these hardcoded; we patch them here)
cd "$EXPERIMENTS_DIR"

CHECKPOINT_DIR="$EXPERIMENTS_DIR/../checkpoints/$EXP_DATE_BASE"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory already exists: $CHECKPOINT_DIR"
    echo "       Choose a different EXP_DATE_BASE or remove the old directory."
    exit 1
fi

python - <<PYEOF
import pandas as pd
import os, shutil, sys
from datetime import datetime
from zoneinfo import ZoneInfo

LA = ZoneInfo("America/Los_Angeles")

# Config from environment
ITERS = $ITERS
BREADTH = $BREADTH
TOPK_CANDIDATES = $TOPK_CANDIDATES
NUM_SAMPLES = $NUM_SAMPLES
MAX_THRESHOLD = $MAX_THRESHOLD
MIN_THRESHOLD = $MIN_THRESHOLD
TOPK = $TOPK
EXP_N = $EXP_N
REL_TOL = $REL_TOL
PROFILE_MODE = "$PROFILE_MODE"
project_name = "$PROJECT_NAME"
org_name = "$ORG_NAME"
exp_date_base = "$EXP_DATE_BASE"

proxy_problem_list_df = pd.read_csv("candidates.csv")
proxy_profile_results_df = pd.read_csv("profile_results.csv")

exp_base_dir = os.path.abspath(f"../checkpoints/{exp_date_base}")
ACCELOPT_BASE_DIR = os.environ["ACCELOPT_BASE_DIR"]
single_loop_exec = os.path.join(ACCELOPT_BASE_DIR, "templates", "complete_local", "run_single_loop.sh")

first_exp_date = datetime.now(LA).strftime("%m-%d-%H-%M")

SUBSTITUTIONS = {
    15: lambda _i, _r: str(EXP_N),
    14: lambda _i, _r: str(TOPK),
    13: lambda _i, _r: str(MIN_THRESHOLD),
    12: lambda _i, _r: str(MAX_THRESHOLD),
    11: lambda _i, _r: str(NUM_SAMPLES),
    10: lambda _i, _r: str(TOPK_CANDIDATES),
    9:  lambda _i, _r: str(BREADTH),
    8:  lambda _i, _r: str(ITERS),
    7:  lambda _i, _r: f'"{org_name}"',
    6:  lambda _i, _r: str(REL_TOL),
    5:  lambda _i, _r: f'"{project_name}"',
    4:  lambda _i, _r: PROFILE_MODE,
    3:  lambda _i, _r: f'"{_r["eval_prefix"]}"',
    2:  lambda _i, _r: f'"{_r["eval_first_exp_date"]}"',
    1:  lambda _i, _r: f'"{_r["new_exp_base_dir"]}"',
}

for index, row in proxy_problem_list_df.iterrows():
    service_name = row["service_name"]
    new_exp_base_dir = os.path.join(exp_base_dir, service_name)
    os.makedirs(new_exp_base_dir, exist_ok=False)
    new_exp_config_dir = os.path.join(new_exp_base_dir, "configs")
    shutil.copytree("./configs", new_exp_config_dir, dirs_exist_ok=True)

    eval_prefix = f"eval-{index}-{exp_date_base}"
    eval_first_exp_date = f"{eval_prefix}-{first_exp_date}"
    new_exp_candidates_dir = os.path.join(new_exp_base_dir, eval_first_exp_date, "candidates")
    os.makedirs(new_exp_candidates_dir, exist_ok=False)

    profile_results_df = proxy_profile_results_df[proxy_profile_results_df["service_name"] == service_name]
    profile_results_df.to_csv(os.path.join(new_exp_candidates_dir, "profile_results.csv"), index=False)

    cur_single_loop_exec_path = os.path.join(exp_base_dir, f"run_single_loop_{service_name}.sh")

    row_ctx = {
        "new_exp_base_dir": new_exp_base_dir,
        "eval_first_exp_date": eval_first_exp_date,
        "eval_prefix": eval_prefix,
    }

    with open(single_loop_exec, "r") as f:
        content = f.read()
    for n in sorted(SUBSTITUTIONS.keys(), reverse=True):
        content = content.replace(f"\${n}", SUBSTITUTIONS[n](index, row_ctx))
    with open(cur_single_loop_exec_path, "w") as f:
        f.write(content)

    print(f"  Created: {service_name}")
    print(f"    Dir:    {new_exp_base_dir}")
    print(f"    Script: {cur_single_loop_exec_path}")

print(f"\nScaffolded {len(proxy_problem_list_df)} problems under {exp_base_dir}")
PYEOF

# =============================================================================
# Step 3: Launch the optimization loops
# =============================================================================
CHECKPOINT_DIR="$(realpath "$EXPERIMENTS_DIR/../checkpoints/$EXP_DATE_BASE")"

echo ""
echo ">>> Generated run scripts:"
for script in "$CHECKPOINT_DIR"/run_single_loop_*.sh; do
    echo "    $script"
done

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo ">>> DRY_RUN=true — stopping before execution."
    echo "    To run a single problem:  bash <script_path>"
    echo "    To run all problems:      DRY_RUN=false bash run_experiment.sh"
    exit 0
fi

echo ""
echo ">>> Launching optimization loops (sequential per problem)..."
echo "    Press Ctrl+C to stop."
echo ""

for script in "$CHECKPOINT_DIR"/run_single_loop_*.sh; do
    problem_name=$(basename "$script" .sh | sed 's/run_single_loop_//')
    echo "=== Starting: $problem_name ==="
    bash "$script" 2>&1 | tee "$CHECKPOINT_DIR/${problem_name}.log"
    echo "=== Finished: $problem_name ==="
    echo ""
done

echo ">>> All problems complete. Results in: $CHECKPOINT_DIR"
