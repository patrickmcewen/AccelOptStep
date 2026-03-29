"""Pipeline registry — single source of truth for composable pipeline configuration.

A pipeline string like 'pytorch-step' or 'numpy-step-nki' is parsed into
frontend, optional middleend, and backend components. Each component maps to
concrete configuration (bench dir, prompt paths, profiler, etc.).

The BACKEND determines which benchmarks to load, which profiler to use, and
what the baselines look like. The FRONTEND determines how problems are
described to the LLM (pytorch nn.Module vs numpy forward()).
"""

PIPELINES = {
    "pytorch-step": {"frontend": "pytorch", "middleend": None, "backend": "step"},
    "numpy-nki": {"frontend": "numpy", "middleend": None, "backend": "nki"},
    "pytorch-nki": {"frontend": "pytorch", "middleend": None, "backend": "nki"},
    "numpy-step": {"frontend": "numpy", "middleend": None, "backend": "step"},
    "pytorch-step-nki": {"frontend": "pytorch", "middleend": "step", "backend": "nki"},
    "numpy-step-nki": {"frontend": "numpy", "middleend": "step", "backend": "nki"},
}

FRONTEND_CONFIG = {
    "pytorch": {
        "description": "PyTorch nn.Module problem specifications",
    },
    "numpy": {
        "description": "NumPy forward() problem specifications",
    },
}

BACKEND_CONFIG = {
    "step": {
        "bench_dir": "StepBench",
        "problem_key": "problem",
        "baseline_key": "baseline",
        "prompts_subdir": "backends/step",
        "profiler": "step",
        "speedup_metric": "cycles",
        "needs_machine_config": True,
        "code_preamble": "step",
    },
    "nki": {
        "bench_dir": "NKIBench",
        "problem_key": "task",
        "baseline_key": "kernel",
        "prompts_subdir": "backends/nki",
        "profiler": "nki",
        "speedup_metric": "latency",
        "needs_machine_config": False,
        "code_preamble": "nki",
    },
}

# For backward compatibility: map old benchmark_type values to pipeline strings
_BENCHMARK_TYPE_COMPAT = {
    "step": "pytorch-step",
    "nki": "numpy-nki",
}


def resolve_pipeline(pipeline_or_benchmark_type: str) -> dict:
    """Resolve a pipeline string (or legacy benchmark_type) into a flat config dict.

    Returns dict with keys:
        pipeline, frontend, middleend, backend,
        bench_dir, problem_key, baseline_key,
        prompts_subdir, profiler, speedup_metric, needs_machine_config, code_preamble
    """
    # Backward compat: translate old benchmark_type values
    pipeline_str = _BENCHMARK_TYPE_COMPAT.get(pipeline_or_benchmark_type, pipeline_or_benchmark_type)

    assert pipeline_str in PIPELINES, (
        f"Unknown pipeline '{pipeline_str}'. "
        f"Valid pipelines: {list(PIPELINES.keys())}"
    )

    components = PIPELINES[pipeline_str]
    frontend = components["frontend"]
    backend = components["backend"]

    be = BACKEND_CONFIG[backend]

    return {
        "pipeline": pipeline_str,
        "frontend": frontend,
        "middleend": components["middleend"],
        "backend": backend,
        # Backend-derived (bench, profiler, baselines all come from backend)
        "bench_dir": be["bench_dir"],
        "problem_key": be["problem_key"],
        "baseline_key": be["baseline_key"],
        "prompts_subdir": be["prompts_subdir"],
        "profiler": be["profiler"],
        "speedup_metric": be["speedup_metric"],
        "needs_machine_config": be["needs_machine_config"],
        "code_preamble": be["code_preamble"],
    }
