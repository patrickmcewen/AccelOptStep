#!/usr/bin/env bash
# AccelOptStep environment setup.
#
# Usage:
#   source env.sh          # Find running step_artifact container, sync code, enter shell
#   source env.sh --sync   # Just sync code + set vars (no interactive shell)
#   source env.sh --run "python3 tests/test_correctness_small.py --all --small"
#
# Inside the container, PYTHONPATH and working directory are set automatically.

set -euo pipefail
# Undo set -e before returning to interactive shell (sourced scripts
# propagate shell options to the caller, which kills the session on any error).
trap 'set +eu' RETURN 2>/dev/null || true

DOCKER_IMAGE="step_artifact"
HOST_DIR="/home/ubuntu/patrick/AbstractOpt/AccelOptStep"
CONTAINER_DIR="/root/AccelOptStep"
STEP_SRC="/root/step_artifact/src"
PYTHONPATH_INSIDE="${STEP_SRC}:${STEP_SRC}/step_py:${STEP_SRC}/sim:${STEP_SRC}/proto"

# --- Find or start the container ---
CONTAINER_ID=$(docker ps -q --filter "ancestor=${DOCKER_IMAGE}" | head -1)

if [ -z "${CONTAINER_ID}" ]; then
    echo "No running ${DOCKER_IMAGE} container found. Starting one..."
    CONTAINER_ID=$(docker run -dit -v "${HOST_DIR}:${CONTAINER_DIR}" "${DOCKER_IMAGE}" bash)
    echo "Started container: ${CONTAINER_ID}"

    # First-time setup: install Python deps and build step_perf
    echo "Installing Python dependencies..."
    docker exec "${CONTAINER_ID}" bash -c \
        'pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu numpy protobuf "maturin[patchelf]" 2>&1 | tail -3'

    echo "Building step_perf Rust simulator..."
    docker exec "${CONTAINER_ID}" bash -c \
        'cd /root/step_artifact/step-perf && maturin build --release 2>&1 | tail -3 && pip install --break-system-packages target/wheels/*.whl 2>&1 | tail -3'
else
    echo "Using existing container: ${CONTAINER_ID}"
fi

# --- Sync code into container ---
# If the container has a bind mount, the host dir IS the container dir — no sync needed.
# Otherwise, copy the code in.
MOUNT_CHECK=$(docker inspect "${CONTAINER_ID}" --format '{{range .Mounts}}{{.Destination}}{{end}}')
if echo "${MOUNT_CHECK}" | grep -q "${CONTAINER_DIR}"; then
    echo "Bind mount detected — host dir is live inside container, skipping sync."
else
    echo "Syncing ${HOST_DIR} -> ${CONTAINER_ID}:${CONTAINER_DIR}"
    docker exec "${CONTAINER_ID}" rm -rf "${CONTAINER_DIR}"
    docker cp "${HOST_DIR}" "${CONTAINER_ID}:${CONTAINER_DIR}"
fi

# --- Export for use by other scripts ---
export ACCELOPTSTEP_CONTAINER="${CONTAINER_ID}"
export ACCELOPTSTEP_PYTHONPATH="${PYTHONPATH_INSIDE}"

# Helper: run a command inside the container with the right env
acceloptstep_exec() {
    docker exec "${ACCELOPTSTEP_CONTAINER}" bash -c \
        "cd ${CONTAINER_DIR} && PYTHONPATH=${PYTHONPATH_INSIDE}:\$PYTHONPATH $*"
}
export -f acceloptstep_exec

# --- Handle mode ---
MODE="${1:-}"

case "${MODE}" in
    --run)
        shift
        CMD="$*"
        if [ -z "${CMD}" ]; then echo "Usage: env.sh --run \"<command>\""; exit 1; fi
        echo "Running: ${CMD}"
        acceloptstep_exec "${CMD}"
        ;;
    --sync)
        echo "Synced. Use acceloptstep_exec \"<cmd>\" to run commands in the container."
        ;;
    *)
        echo "Entering interactive shell in container. PYTHONPATH is set."
        echo "  Working dir: ${CONTAINER_DIR}"
        echo "  Detach with Ctrl+p, Ctrl+q (do NOT type 'exit')"
        docker exec -it "${CONTAINER_ID}" bash -c \
            "export PYTHONPATH=${PYTHONPATH_INSIDE}:\$PYTHONPATH && cd ${CONTAINER_DIR} && exec bash"
        ;;
esac
