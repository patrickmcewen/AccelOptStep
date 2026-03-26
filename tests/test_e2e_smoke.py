"""End-to-end smoke test: baseline → profile → prompt formatting."""
import json
import sys
import os

sys.path.insert(0, os.environ.get("STEP_ARTIFACT_SRC", "/home/ubuntu/patrick/AbstractOpt/step_artifact/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_baseline_profile_to_prompt():
    """Load GEMM baseline, profile it symbolically, format for planner prompt."""
    # Step 1: Read the baseline code
    baseline_path = os.path.join(os.path.dirname(__file__), "../StepBench/baselines/gemm_square.py")
    with open(baseline_path) as f:
        code = f.read()

    # Step 2: Profile it
    from accelopt.step_kernel_wrapper import StepKernel, ProfileMode

    kernel = StepKernel(
        step_code=code,
        problem_path="StepBench/problems/gemm_square.py",
        profile_mode=ProfileMode.SYMBOLIC,
    )
    result = kernel.profile()
    assert result.compiled, "Kernel should compile"
    assert result.runnable, "Kernel should be runnable"
    assert "off_chip_bytes" in result.metadata, f"Missing off_chip_bytes in {result.metadata}"
    assert "on_chip_bytes" in result.metadata, f"Missing on_chip_bytes in {result.metadata}"

    # Step 3: Format for prompt
    profile_str = json.dumps(result.metadata, indent=2, default=str)
    assert "off_chip_bytes" in profile_str

    # Step 4: Verify displayed_profiles.json exists and is valid
    displayed_path = os.path.join(
        os.path.dirname(__file__), "../prompts/planner_prompts/displayed_profiles.json"
    )
    with open(displayed_path) as f:
        displayed = json.load(f)
    assert "off_chip_bytes" in displayed

    print(f"Profile result: {profile_str}")
    print("Smoke test PASSED")


def test_gemm_silu_baseline_profiles():
    """Verify fused GEMM+SiLU baseline can be profiled."""
    baseline_path = os.path.join(os.path.dirname(__file__), "../StepBench/baselines/gemm_silu.py")
    with open(baseline_path) as f:
        code = f.read()

    from accelopt.step_kernel_wrapper import StepKernel, ProfileMode

    kernel = StepKernel(
        step_code=code,
        problem_path="StepBench/problems/gemm_silu.py",
        profile_mode=ProfileMode.SYMBOLIC,
    )
    result = kernel.profile()
    assert result.compiled
    assert result.runnable
    assert "off_chip_bytes" in result.metadata


def test_sdpa_baseline_profiles():
    """Verify SDPA baseline can be profiled."""
    baseline_path = os.path.join(os.path.dirname(__file__), "../StepBench/baselines/sdpa.py")
    with open(baseline_path) as f:
        code = f.read()

    from accelopt.step_kernel_wrapper import StepKernel, ProfileMode

    kernel = StepKernel(
        step_code=code,
        problem_path="StepBench/problems/sdpa.py",
        profile_mode=ProfileMode.SYMBOLIC,
    )
    result = kernel.profile()
    assert result.compiled
    assert result.runnable
    assert "off_chip_bytes" in result.metadata
