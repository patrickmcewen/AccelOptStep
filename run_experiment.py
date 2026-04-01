#!/usr/bin/env python3
"""AccelOptStep — Run the agentic optimization loop (pytorch-step pipeline).

Usage:
    python3 run_experiment.py --config <preset_name>

Prerequisites:
    - vLLM server running at http://localhost:31001/v1 (or update configs/)
    - Docker container with step_artifact image
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

from src.bootstrap import load_config, setup_environment
from src.orchestrator import (
    generate_profile_csv,
    launch_loops,
    resume_experiment,
    scaffold_experiments,
)


def main():
    parser = argparse.ArgumentParser(description="AccelOptStep — Run the agentic optimization loop")
    parser.add_argument("--config_file", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--config", required=True, help="Named preset from the config file (e.g. full_run, test_small_scale_run)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an existing checkpoint directory to resume from (skips profiling and scaffolding)")
    args = parser.parse_args()

    cfg = load_config(args.config_file, args.config)
    script_dir = Path(__file__).resolve().parent

    machine_config_preset = cfg.get("machine_config", "default")
    machine_config_path = str(script_dir / "machine_config.yaml")

    setup_environment(script_dir, cfg)

    if args.resume:
        checkpoint_dir = Path(args.resume).resolve()
        assert checkpoint_dir.exists(), f"Checkpoint directory does not exist: {checkpoint_dir}"
        resume_experiment(checkpoint_dir, cfg, script_dir,
                          machine_config_path=machine_config_path,
                          machine_config_preset=machine_config_preset)
        return

    # Resolve exp_date_base (auto-generate if not in config)
    exp_date_base = cfg.get("exp_date_base") or datetime.now().strftime("%Y-%m-%d-%H%M%S")

    configs_dir = script_dir / "configs"

    # Create the run-local checkpoint directory up-front so all generated
    # artifacts (candidates.csv, profile_results.csv) are written here
    # instead of the shared experiments_dir — avoids race conditions when
    # multiple flows run concurrently.
    checkpoint_dir = (script_dir / "checkpoints" / exp_date_base).resolve()
    assert not checkpoint_dir.exists(), (
        f"Checkpoint directory already exists: {checkpoint_dir}\n"
        f"Choose a different exp_date_base or remove the old directory."
    )
    os.makedirs(checkpoint_dir)

    generate_profile_csv(script_dir, checkpoint_dir, cfg["profile_mode"], cfg,
                         machine_config_path=machine_config_path, machine_config_preset=machine_config_preset)
    problem_configs = scaffold_experiments(script_dir, checkpoint_dir, configs_dir, cfg, exp_date_base,
                                           machine_config_path=machine_config_path,
                                           machine_config_preset=machine_config_preset)
    launch_loops(checkpoint_dir, problem_configs, cfg.get("dry_run", False))


if __name__ == "__main__":
    main()
