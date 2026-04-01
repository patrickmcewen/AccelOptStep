"""Pipeline config — hardcoded for pytorch-step (the only supported pipeline).

Other modules import PIPELINE_CONFIG to get bench_dir, profiler, speedup_metric, etc.
"""

PIPELINE_CONFIG = {
    "pipeline": "pytorch-step",
    "frontend": "pytorch",
    "backend": "step",
    "bench_dir": "StepBench",
    "problem_key": "problem",
    "baseline_key": "baseline",
    "profiler": "step",
    "speedup_metric": "cycles",
    "needs_machine_config": True,
    "code_preamble": "step",
}
