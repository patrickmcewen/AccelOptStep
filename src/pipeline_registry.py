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

MIDDLEEND_CONFIG = {
    "step": {
        "prompts_subdir": "middleends/step",
        "profiler": "step",
        "speedup_metric": "cycles",
        "needs_machine_config": True,
        "code_preamble": "step",
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

    middleend = components["middleend"]

    result = {
        "pipeline": pipeline_str,
        "frontend": frontend,
        "middleend": middleend,
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

    # Add middleend-specific config when present
    if middleend is not None:
        assert middleend in MIDDLEEND_CONFIG, (
            f"Unknown middleend '{middleend}'. "
            f"Valid middleends: {list(MIDDLEEND_CONFIG.keys())}"
        )
        me = MIDDLEEND_CONFIG[middleend]
        result["middleend_prompts_subdir"] = me["prompts_subdir"]
        result["middleend_profiler"] = me["profiler"]
        result["middleend_speedup_metric"] = me["speedup_metric"]
        result["middleend_needs_machine_config"] = me["needs_machine_config"]
        result["middleend_code_preamble"] = me["code_preamble"]

    return result


def get_stage_configs(pipeline_str: str) -> tuple[dict | None, dict]:
    """For multi-stage pipelines, return (middleend_stage_config, backend_stage_config).

    For single-stage pipelines, return (None, {}) — no overrides needed.

    Each stage_config is a dict of overrides to apply on top of the resolved pipeline.
    Stage 1 (middleend) overrides prompts/profiler to use the middleend.
    Stage 2 (backend) uses the default resolved config (no overrides).
    """
    cfg = resolve_pipeline(pipeline_str)
    if cfg["middleend"] is None:
        return None, {}

    me = MIDDLEEND_CONFIG[cfg["middleend"]]
    stage1_overrides = {
        "prompts_subdir": me["prompts_subdir"],
        "profiler": me["profiler"],
        "speedup_metric": me["speedup_metric"],
        "needs_machine_config": me["needs_machine_config"],
        "code_preamble": me["code_preamble"],
    }
    # Stage 2 uses the default backend config — no overrides
    return stage1_overrides, {}
