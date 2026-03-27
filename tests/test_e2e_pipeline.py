"""End-to-end smoke tests for the AccelOptStep pipeline data flow.

Verifies candidate collection, prompt construction, and bench_config.yaml format
without requiring an LLM API key or Docker.
"""

import json
import os
import subprocess
import tempfile

import pandas as pd

ACCELOPT_BASE_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
ACCELOPT_BASE_DIR = os.path.abspath(ACCELOPT_BASE_DIR)

STEP_ARTIFACT_SRC = "/home/ubuntu/patrick/AbstractOpt/step_artifact/src"

ENV = {
    **os.environ,
    "ACCELOPT_BASE_DIR": ACCELOPT_BASE_DIR,
    "PYTHONPATH": f"{STEP_ARTIFACT_SRC}:{os.environ.get('PYTHONPATH', '')}",
}


def test_collect_candidates_generates_valid_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        candidates_path = os.path.join(tmpdir, "candidates.csv")
        profile_path = os.path.join(tmpdir, "profile.csv")

        result = subprocess.run(
            [
                "python", os.path.join(ACCELOPT_BASE_DIR, "scripts", "collect_candidates.py"),
                "--mode", "collect",
                "--output_candidates_path", candidates_path,
                "--output_profile_path", profile_path,
            ],
            env=ENV,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"collect_candidates.py failed:\n{result.stderr}"

        df = pd.read_csv(candidates_path)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

        required_columns = ["problem", "values", "case_id", "task", "kernel", "case_name", "service_name"]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        for _, row in df.iterrows():
            assert os.path.isfile(row["task"]), f"Task file not found: {row['task']}"
            assert os.path.isfile(row["kernel"]), f"Kernel file not found: {row['kernel']}"


def test_planner_prompt_construction():
    base_prompt_path = os.path.join(ACCELOPT_BASE_DIR, "prompts", "planner_prompts", "base_prompt.txt")
    empty_rewrites_path = os.path.join(ACCELOPT_BASE_DIR, "prompts", "empty_rewrites.json")
    script_path = os.path.join(ACCELOPT_BASE_DIR, "prompts", "planner_prompts", "construct_base_prompt.py")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "constructed_prompt.txt")

        result = subprocess.run(
            [
                "python", script_path,
                "--original_base_prompt_path", base_prompt_path,
                "--summarizer_output_list_path", empty_rewrites_path,
                "--new_base_prompt_path", output_path,
            ],
            env=ENV,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"construct_base_prompt.py failed:\n{result.stderr}"

        with open(output_path, "r") as f:
            prompt_text = f.read()
        assert len(prompt_text) > 100, f"Prompt too short ({len(prompt_text)} chars)"


def test_bench_config_format():
    from StepBench.loader import load_config
    config = load_config()

    assert len(config) > 0, "bench_config.yaml is empty"

    for bench_name, bench_info in config.items():
        assert "problem" in bench_info, f"{bench_name} missing 'problem'"
        assert "params" in bench_info, f"{bench_name} missing 'params'"
        assert "presets" in bench_info, f"{bench_name} missing 'presets'"
        assert isinstance(bench_info["presets"], dict), f"{bench_name} 'presets' is not a dict"
        assert len(bench_info["presets"]) > 0, f"{bench_name} has no presets"

        for preset_name, preset_dims in bench_info["presets"].items():
            assert isinstance(preset_dims, dict), f"{bench_name}/{preset_name} dims is not a dict"
            for param in bench_info["params"]:
                assert param in preset_dims, f"{bench_name}/{preset_name} missing param '{param}'"
