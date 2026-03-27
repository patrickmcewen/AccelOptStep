"""Run a non-first iteration of the experiment loop (has prior rewrites)."""

import argparse
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
    project_name: str,
    org_name: str,
    logfire_enabled: bool = True,
    log_file: Path = None,
) -> None:
    base_dir = Path(os.environ["ACCELOPT_BASE_DIR"])
    exp_dir = exp_base_dir / exp_date

    assert exp_dir.exists(), f"Experiment directory does not exist: {exp_dir}"

    # Logfire setup (read existing env name from file)
    if logfire_enabled:
        subprocess.run(
            ["logfire", "projects", "use", project_name, "--org", org_name],
            cwd=str(exp_dir),
            check=True,
        )
    logfire_env_name = (exp_dir / "logfire_env_name.txt").read_text().strip()
    os.environ["LOGFIRE_ENVIRONMENT"] = logfire_env_name
    print(f"LOGFIRE_ENVIRONMENT: {logfire_env_name}")

    # Construct experience (with prior rewrites)
    construct_experience_exec = base_dir / "scripts" / "construct_experience.py"
    construct_experience_output_path = exp_dir / "rewrites" / "aggregated_rewrites_list.json"
    original_rewrite_list_path = exp_dir / "rewrites" / "rewrites_selection_output_list.json"
    subprocess.run(
        [
            sys.executable, str(construct_experience_exec),
            "--original_rewrite_list_path", str(original_rewrite_list_path),
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
    planner_prompt_constructor_exec = base_dir / "prompts" / "planner_prompts" / "construct_base_prompt.py"
    original_base_prompt_path = base_dir / "prompts" / "planner_prompts" / "base_prompt.txt"
    new_base_prompt_path = exp_dir / "planner_prompts" / "base_prompt.txt"
    subprocess.run(
        [
            sys.executable, str(planner_prompt_constructor_exec),
            "--original_base_prompt_path", str(original_base_prompt_path),
            "--summarizer_output_list_path", str(construct_experience_output_path),
            "--new_base_prompt_path", str(new_base_prompt_path),
            "--log_file", str(log_file),
        ],
        cwd=str(exp_dir),
        check=True,
    )

    # Planner
    planner_exec = base_dir / "scripts" / "planner.py"
    planner_output_path = exp_dir / "planner_results.json"
    planner_user_template_path = base_dir / "prompts" / "planner_prompts" / "planner_prompt_template.txt"
    planner_profile_result_path = exp_dir / "candidates" / "profile_results.csv"
    planner_model_config_path = exp_base_dir / "configs" / "planner_config.json"
    planner_displayed_profiles_path = base_dir / "prompts" / "planner_prompts" / "displayed_profiles.json"
    subprocess.run(
        [
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
        ],
        cwd=str(exp_dir),
        check=True,
    )

    # Executor
    executor_exec = base_dir / "scripts" / "executor.py"
    executor_base_prompt_path = base_dir / "prompts" / "executor_prompts" / "base_prompt.txt"
    executor_user_template_path = base_dir / "prompts" / "executor_prompts" / "user_prompt_template.txt"
    executor_model_config_path = exp_base_dir / "configs" / "executor_config.json"
    executor_log_output_path = exp_dir / "executor_results.json"
    subprocess.run(
        [
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
        ],
        cwd=str(exp_dir),
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run non-first iteration of experiment loop")
    parser.add_argument("exp_date", type=str)
    parser.add_argument("exp_base_dir", type=Path)
    parser.add_argument("experience_list_path", type=Path)
    parser.add_argument("profile_mode", type=str)
    parser.add_argument("breadth", type=int)
    parser.add_argument("num_samples", type=int)
    parser.add_argument("exp_n", type=int)
    parser.add_argument("project_name", type=str)
    parser.add_argument("org_name", type=str)
    parser.add_argument("log_file", type=Path)
    args = parser.parse_args()

    run(
        exp_date=args.exp_date,
        exp_base_dir=args.exp_base_dir,
        experience_list_path=args.experience_list_path,
        profile_mode=args.profile_mode,
        breadth=args.breadth,
        num_samples=args.num_samples,
        exp_n=args.exp_n,
        project_name=args.project_name,
        org_name=args.org_name,
        log_file=args.log_file,
    )
