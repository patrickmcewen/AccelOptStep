"""Run the first iteration of the experiment loop (no prior rewrites)."""

import os
import subprocess
import sys
from pathlib import Path


def run(
    exp_date: str,
    exp_base_dir: Path,
    experience_list_path: Path,
    profile_mode: str,
    breadth: int,
    num_samples: int,
    exp_n: int,
    rel_tol: float,
    project_name: str,
    org_name: str,
    logfire_enabled: bool = True,
    log_file: Path = None,
    machine_config_path: str = None,
    machine_config_preset: str = "default",
    include_baseline: bool = False,
) -> None:
    base_dir = Path(os.environ["ACCELOPT_BASE_DIR"])
    prompts_base = base_dir / "prompts"
    exp_dir = exp_base_dir / exp_date

    assert exp_dir.exists(), f"Experiment directory does not exist: {exp_dir}"

    # Logfire setup
    logfire_env_name = exp_date.replace("-", "")
    if logfire_enabled:
        result = subprocess.run(
            ["logfire", "projects", "use", project_name, "--org", org_name],
            cwd=str(exp_dir),
        )
        if result.returncode != 0:
            print(f"WARNING: logfire setup failed (exit {result.returncode}), continuing without telemetry")
    (exp_dir / "logfire_env_name.txt").write_text(logfire_env_name)
    os.environ["LOGFIRE_ENVIRONMENT"] = logfire_env_name
    print(f"LOGFIRE_ENVIRONMENT: {logfire_env_name}")

    # Construct experience
    (exp_dir / "rewrites").mkdir(parents=True, exist_ok=True)
    construct_experience_exec = base_dir / "src" / "agents" / "construct_experience.py"
    construct_experience_output_path = exp_dir / "rewrites" / "aggregated_rewrites_list.json"
    subprocess.run(
        [
            sys.executable, str(construct_experience_exec),
            "--is_first",
            "--experience_list_path", str(experience_list_path),
            "--output_path", str(construct_experience_output_path),
            "--n", str(exp_n),
            "--log_file", str(log_file),
        ],
        cwd=str(exp_dir),
        check=True,
    )

    # Construct planner base prompt
    (exp_dir / "planner_prompts").mkdir(parents=True, exist_ok=True)
    planner_prompt_constructor_exec = prompts_base / "planner_prompts" / "construct_base_prompt.py"
    original_base_prompt_path = prompts_base / "planner_prompts" / "base_prompt.txt"
    new_base_prompt_path = exp_dir / "planner_prompts" / "base_prompt.txt"
    construct_prompt_cmd = [
        sys.executable, str(planner_prompt_constructor_exec),
        "--original_base_prompt_path", str(original_base_prompt_path),
        "--summarizer_output_list_path", str(construct_experience_output_path),
        "--new_base_prompt_path", str(new_base_prompt_path),
        "--log_file", str(log_file),
        "--machine_config_preset", machine_config_preset,
    ]
    if machine_config_path:
        construct_prompt_cmd += ["--machine_config_path", machine_config_path]
    subprocess.run(construct_prompt_cmd, cwd=str(exp_dir), check=True)

    # Planner
    planner_exec = base_dir / "src" / "agents" / "planner.py"
    planner_output_path = exp_dir / "planner_results.json"
    planner_user_template_path = prompts_base / "planner_prompts" / "planner_prompt_template.txt"
    planner_profile_result_path = exp_dir / "candidates" / "profile_results.csv"
    planner_model_config_path = exp_base_dir / "configs" / "planner_config.json"
    planner_displayed_profiles_path = prompts_base / "planner_prompts" / "displayed_profiles.json"
    planner_cmd = [
        sys.executable, str(planner_exec),
        "--output_path", str(planner_output_path),
        "--breadth", str(breadth),
        "--exp_dir", str(exp_dir),
        "--base_prompt_path", str(new_base_prompt_path),
        "--user_template_path", str(planner_user_template_path),
        "--profile_result_path", str(planner_profile_result_path),
        "--model_config_path", str(planner_model_config_path),
        "--displayed_profiles_path", str(planner_displayed_profiles_path),
        "--log_file", str(log_file),
        "--machine_config_preset", machine_config_preset,
    ]
    if machine_config_path:
        planner_cmd += ["--machine_config_path", machine_config_path]
    if include_baseline:
        planner_cmd += ["--include_baseline"]
    subprocess.run(planner_cmd, cwd=str(exp_dir), check=True)

    # Executor
    executor_exec = base_dir / "src" / "agents" / "executor.py"
    executor_base_prompt_path = prompts_base / "executor_prompts" / "base_prompt.txt"
    executor_user_template_path = prompts_base / "executor_prompts" / "user_prompt_template.txt"
    executor_model_config_path = exp_base_dir / "configs" / "executor_config.json"
    executor_log_output_path = exp_dir / "executor_results.json"
    executor_cmd = [
        sys.executable, str(executor_exec),
        "--num_samples", str(num_samples),
        "--problems_path", str(planner_profile_result_path),
        "--extractor_output_path", str(planner_output_path),
        "--exp_dir", str(exp_dir),
        "--base_prompt_path", str(executor_base_prompt_path),
        "--user_template_path", str(executor_user_template_path),
        "--model_config_path", str(executor_model_config_path),
        "--profile_mode", profile_mode,
        "--output_path", str(executor_log_output_path),
        "--exp_date", exp_date,
        "--log_file", str(log_file),
        "--machine_config_preset", machine_config_preset,
        "--rel_tol", str(rel_tol),
    ]
    if machine_config_path:
        executor_cmd += ["--machine_config_path", machine_config_path]
    if include_baseline:
        executor_cmd += ["--include_baseline"]
    subprocess.run(
        executor_cmd,
        cwd=str(exp_dir),
        check=True,
    )
