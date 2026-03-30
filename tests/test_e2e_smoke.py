"""End-to-end smoke test: baseline + loader -> profile -> prompt formatting."""
import json
import sys
import os

sys.path.insert(0, os.environ.get("STEP_ARTIFACT_SRC", "/home/ubuntu/patrick/AbstractOpt/step_artifact/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from StepBench.loader import get_dims


def test_baseline_profile_to_prompt():
    baseline_path = os.path.join(os.path.dirname(__file__), "../StepBench/baselines/gemm.py")
    with open(baseline_path) as f:
        code = f.read()

    from src.step_kernel_wrapper import StepKernel, ProfileMode

    dims = get_dims("gemm", "small")
    kernel = StepKernel(
        step_code=code,
        problem_path="StepBench/problems/gemm.py",
        profile_mode=ProfileMode.SYMBOLIC,
        dims=dims,
    )
    result = kernel.profile()
    assert result.compiled, "Kernel should compile"
    assert result.runnable, "Kernel should be runnable"
    assert "off_chip_bytes" in result.metadata

    profile_str = json.dumps(result.metadata, indent=2, default=str)
    assert "off_chip_bytes" in profile_str

    displayed_path = os.path.join(
        os.path.dirname(__file__), "../prompts/planner_prompts/displayed_profiles.json"
    )
    with open(displayed_path) as f:
        displayed = json.load(f)
    assert "off_chip_bytes" in displayed

    print(f"Profile result: {profile_str}")
    print("Smoke test PASSED")


def test_gemm_swish_scaling_baseline_profiles():
    baseline_path = os.path.join(os.path.dirname(__file__), "../StepBench/baselines/gemm_swish_scaling.py")
    with open(baseline_path) as f:
        code = f.read()

    from src.step_kernel_wrapper import StepKernel, ProfileMode

    dims = get_dims("gemm_swish_scaling", "small")
    kernel = StepKernel(
        step_code=code,
        problem_path="StepBench/problems/gemm_swish_scaling.py",
        profile_mode=ProfileMode.SYMBOLIC,
        dims=dims,
    )
    result = kernel.profile()
    assert result.compiled
    assert result.runnable
    assert "off_chip_bytes" in result.metadata


def test_sdpa_baseline_profiles():
    baseline_path = os.path.join(os.path.dirname(__file__), "../StepBench/baselines/sdpa.py")
    with open(baseline_path) as f:
        code = f.read()

    from src.step_kernel_wrapper import StepKernel, ProfileMode

    dims = get_dims("sdpa", "small")
    kernel = StepKernel(
        step_code=code,
        problem_path="StepBench/problems/sdpa.py",
        profile_mode=ProfileMode.SYMBOLIC,
        dims=dims,
    )
    result = kernel.profile()
    assert result.compiled
    assert result.runnable
    assert "off_chip_bytes" in result.metadata
