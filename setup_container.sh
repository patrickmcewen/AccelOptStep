#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# AccelOptStep — One-time container setup
# =============================================================================
#
# Run this once after entering the Docker container via env.sh.
# It installs all Python dependencies and verifies the environment is ready.
#
# Usage (inside container):
#   bash setup_container.sh
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_SRC="/root/step_artifact/src"

# ── PYTHONPATH ───────────────────────────────────────────────────────────────
export PYTHONPATH="${STEP_SRC}:${STEP_SRC}/step_py:${STEP_SRC}/sim:${STEP_SRC}/proto:${PYTHONPATH:-}"
export ACCELOPT_BASE_DIR="$SCRIPT_DIR"

echo "=== AccelOptStep container setup ==="
echo ""

# ── Build proto and step_perf if missing ─────────────────────────────────────
echo "[1/5] Checking STeP artifact build state..."

if [ ! -f "${STEP_SRC}/proto/datatype_pb2.py" ]; then
    echo "      Compiling protobuf files..."
    PROTO_DIR="/root/step_artifact/step_perf_ir/proto"
    OUT_DIR="${STEP_SRC}/proto"
    mkdir -p "$OUT_DIR"
    touch "$OUT_DIR/__init__.py"
    pip install grpcio-tools 2>&1 | tail -2
    python -m grpc_tools.protoc -I"$PROTO_DIR" --python_out="$OUT_DIR" \
        "$PROTO_DIR"/datatype.proto "$PROTO_DIR"/func.proto \
        "$PROTO_DIR"/graph.proto "$PROTO_DIR"/ops.proto
    echo "      Proto compiled."
else
    echo "      Proto files already built."
fi

if ! python -c "import step_perf" 2>/dev/null; then
    echo "      Building step_perf Rust simulator (this takes a minute)..."
    cd /root/step_artifact/step-perf
    pip install "maturin[patchelf]" 2>&1 | tail -2
    maturin build --release 2>&1 | tail -3
    pip install target/wheels/*.whl 2>&1 | tail -2
    cd "$SCRIPT_DIR"
    echo "      step_perf built."
else
    echo "      step_perf already installed."
fi
echo ""

# ── Install accelopt + all Python deps ───────────────────────────────────────
echo "[2/5] Installing accelopt and Python dependencies..."
pip install -e "$SCRIPT_DIR" aiohttp 2>&1 | tail -5
echo "      Done."
echo ""

# ── Verify critical imports ──────────────────────────────────────────────────
echo "[3/5] Verifying imports..."
python -c "
import sys
failures = []
for mod in ['agents', 'openai', 'logfire', 'pandas', 'pydantic', 'torch',
            'sympy', 'networkx', 'aiohttp', 'accelopt', 'proto', 'step_perf']:
    try:
        __import__(mod)
    except ImportError as e:
        failures.append(f'  {mod}: {e}')

# Verify openai-agents, not the RL agents package
from agents import Agent, Runner, ModelBehaviorError, AsyncOpenAI
from accelopt.step_kernel_wrapper import StepKernel
from accelopt.utils import retry_runner_safer

if failures:
    print('FAILED imports:')
    print('\n'.join(failures))
    sys.exit(1)
else:
    print('      All imports OK.')
"
echo ""

# ── Verify LLM endpoint reachability ────────────────────────────────────────
echo "[4/5] Checking LLM endpoint..."
CONFIG_FILE="$SCRIPT_DIR/experiments/full_complete_local/configs/planner_config.json"
LLM_URL=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['url'])")

# Strip /v1 for base URL, then test /v1/models
BASE_URL="${LLM_URL%/v1}"
if curl -sf --connect-timeout 5 "${BASE_URL}/v1/models" > /dev/null 2>&1; then
    echo "      LLM endpoint reachable at ${BASE_URL}"
else
    echo "      WARNING: Cannot reach LLM endpoint at ${BASE_URL}"
    echo "      Make sure your vLLM server is running and the SSH tunnel is active."
    echo "      Config: $CONFIG_FILE"
fi
echo ""

# ── Print run instructions ───────────────────────────────────────────────────
echo "[5/5] Setup complete. To run the experiment:"
echo ""
echo "  export ACCELOPT_BASE_DIR=\"$SCRIPT_DIR\""
echo "  export PYTHONPATH=\"${STEP_SRC}:\${STEP_SRC}/step_py:\${STEP_SRC}/sim:\${STEP_SRC}/proto:\$PYTHONPATH\""
echo "  cd $SCRIPT_DIR"
echo ""
echo "  # Full run (all baselines, 15 iterations):"
echo "  bash run_experiment.sh"
echo ""
echo "  # Quick test (1 iteration, 2 plans):"
echo "  ITERS=1 BREADTH=2 NUM_SAMPLES=1 bash run_experiment.sh"
echo ""
echo "  # Dry run (scaffold only, no execution):"
echo "  DRY_RUN=true bash run_experiment.sh"
echo ""
echo "  # Custom project name:"
echo "  PROJECT_NAME=my-experiment bash run_experiment.sh"
echo ""
