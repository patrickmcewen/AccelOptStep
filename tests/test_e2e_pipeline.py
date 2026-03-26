"""End-to-end smoke tests for the AccelOptStep pipeline data flow.

Verifies candidate collection, prompt construction, and summary.json format
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


def test_summary_json_format():
    summary_path = os.path.join(ACCELOPT_BASE_DIR, "StepBench", "summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    assert len(summary) > 0, "summary.json is empty"

    for problem_name, problem_info in summary.items():
        assert "cases" in problem_info, f"{problem_name} missing 'cases'"
        assert isinstance(problem_info["cases"], dict), f"{problem_name} 'cases' is not a dict"

        for case_id, case_info in problem_info["cases"].items():
            assert "values" in case_info, f"{problem_name}/{case_id} missing 'values'"
            assert isinstance(case_info["values"], dict), f"{problem_name}/{case_id} 'values' is not a dict"
            assert "impls" in case_info, f"{problem_name}/{case_id} missing 'impls'"
            assert isinstance(case_info["impls"], list), f"{problem_name}/{case_id} 'impls' is not a list"
            assert len(case_info["impls"]) > 0, f"{problem_name}/{case_id} has no impls"

            for impl in case_info["impls"]:
                assert "task" in impl, f"{problem_name}/{case_id} impl missing 'task'"
                assert "kernel" in impl, f"{problem_name}/{case_id} impl missing 'kernel'"
