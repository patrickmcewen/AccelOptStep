# Executor — parallel LLM proposal generation + sequential STeP profiling with fixup

import os
import sys
import json
import tempfile
from pathlib import Path
import traceback
import contextlib
import logging
from datetime import datetime, timezone
import multiprocessing as mp
import time
import asyncio
import pandas as pd
from pydantic import BaseModel
import logfire
from src.utils import extract_first_code, retry_runner_safer
from src.eval_step import StepKernelProperties
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunConfig, ModelSettings

# -------------------------- Logging --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------- STeP Import Preamble --------------------------
STEP_IMPORTS = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx import MultiDiGraph
from step_py.ops import (
    OffChipLoad, OffChipStore, BinaryMap, UnaryMap, BinaryMapAccum,
    Accum, Flatten, Reshape, Promote, ExpandRef,
    Bufferize, Streamify, DynStreamify, RetileStreamify,
    FlatPartition, FlatReassemble, Broadcast, RepeatStatic,
    EagerMerge, Parallelize,
    DynOffChipLoad, RandomOffChipLoad, RandomOffChipStore,
)
from step_py.utility_ops import (
    PrinterContext, ConsumerContext, SelectGen,
    FilterLastTile, MetadataGen, ExpertAddrGen,
)
from step_py.functions import map_fn, map_accum_fn, accum_fn, init_fn
from step_py.datatype import Tile, DynTile, Stream, Float16, Float32, Uint32, Uint64, Bool, MultiHot, Index, Select, Buffer, DynDim
from rewrite.broadcast import infer_broadcast

SEED = 42
torch.manual_seed(SEED)
"""

# -------------------------- Config Models --------------------------
class ExecutorPromptConfig(BaseModel):
    host_problem_path: str = ""
    step_kernel_path: str = ""
    user_template_path: str = ""
    optimization_plan: str = ""
    include_baseline: bool = False
    values_json: str = ""

class ExecutorConfig(BaseModel):
    system_prompt: str = ""
    service_name: str = ""
    kernel_path: str = ""
    task_path: str = ""
    optimization_plan: str = ""
    problem: str = ""
    values: str = ""
    case_name: str = ""
    num_samples: int = 4
    user_template_path: str = ""
    profile_mode: str = "cycle_accurate"
    include_baseline: bool = False

# -------------------------- Helpers --------------------------
def construct_executor_prompt(config: ExecutorPromptConfig, executor_experiences_block: str = "") -> str:
    with open(config.host_problem_path, "r") as f:
        host_problem_function = f.read()
    with open(config.step_kernel_path, "r") as f:
        step_kernel_function = f.read()
    with open(config.user_template_path, "r") as f:
        prompt_template = f.read()
    if config.include_baseline and step_kernel_function:
        baseline_block = (
            "# Baseline STeP IR kernel\n"
            "```\n"
            f"{step_kernel_function}\n"
            "```\n"
        )
    else:
        baseline_block = ""
    # Build preamble showing the LLM the exact module-level variables available
    dims = json.loads(config.values_json)
    preamble_lines = [f"{k} = {v!r}" for k, v in dims.items()]

    user_prompt = (
        prompt_template
        .replace("{problem_code}", host_problem_function)
        .replace("{kernel_code}", step_kernel_function)
        .replace("{baseline_context}", baseline_block)
        .replace("{optimization_plan}", config.optimization_plan)
        .replace("{executor_experiences}", executor_experiences_block)
        .replace("{preamble}", STEP_IMPORTS + "\n" + "\n".join(preamble_lines))
    )
    return user_prompt

def _write_temp_kernel(code: str, baseline_code: str, values_json: str | None = None) -> str:
    fd, temp_path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(STEP_IMPORTS)
        f.write("\n")
        # Inject dimension constants (M, K, N, etc.) so the LLM-generated
        # build_graph() can reference them as module-level variables.
        if values_json:
            dims = json.loads(values_json)
            for k, v in dims.items():
                f.write(f"{k} = {v!r}\n")
            f.write("\n")
        f.write(baseline_code)
        f.write("\n")
        f.write(code)
        f.write("\n")
    return temp_path

# -------------------------- Parallel LLM --------------------------
async def propose_once(name: str, config: ExecutorPromptConfig, agent: Agent, executor_experiences_block: str = ""):
    user_prompt = construct_executor_prompt(config, executor_experiences_block=executor_experiences_block)
    if "claude" in agent.model.model.lower():
        run_config = RunConfig(
            model_settings=ModelSettings(
                temperature=1.0, # Temperature must be 1.0 for reasoning to be enabled
                max_tokens=20000,
                extra_body={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 10000
                    }
                }
            )
        )
    else:
        run_config = None
    result = await retry_runner_safer(agent, user_prompt, run_config=run_config, max_retries=15, delay=10)
    assert result is not None, f"retry_runner_safer returned None for {name}"
    logger.info("[propose_once] %s: final_output type=%s len=%d, new_items=%d",
                name, type(result.final_output).__name__, len(result.final_output) if result.final_output else 0,
                len(result.new_items) if result.new_items else 0)
    for idx, item in enumerate(result.new_items or []):
        logger.info("[propose_once] %s: new_items[%d] type=%s: %s",
                    name, idx, type(item).__name__, str(item)[:500])
    code = extract_first_code(result.final_output, ["python"])
    if not code:
        logger.warning("[propose_once] %s: extract_first_code found no code block. Full response:\n%s",
                       name, result.final_output)
        return None
    return {"name": name, "result": result, "code": code}

async def stage1_gather_proposals(service_name: str, pconfig: ExecutorPromptConfig, base_agent: Agent, num_samples: int, executor_experiences_block: str = ""):
    tasks = []
    for i in range(num_samples):
        agent = Agent(name=f"Executor_{i}", instructions=base_agent.instructions, model=base_agent.model)
        tasks.append(asyncio.create_task(propose_once(f"{service_name}_{i}", pconfig, agent, executor_experiences_block=executor_experiences_block)))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]

def _profile_worker(program_path, problem_path, profile_mode, result_path, dims=None,
                    machine_config_path=None, machine_config_preset="default"):
    """Profile a STeP kernel. Runs in a subprocess for isolation."""
    import json, traceback
    from src.step_kernel_wrapper import StepKernel, ProfileMode

    with open(program_path) as f:
        code = f.read()

    mode = ProfileMode.SYMBOLIC if profile_mode == "symbolic" else ProfileMode.CYCLE_ACCURATE
    kernel = StepKernel(step_code=code, problem_path=problem_path, profile_mode=mode, dims=dims,
                       machine_config_path=machine_config_path, machine_config_preset=machine_config_preset)
    result = kernel.profile()

    with open(result_path, "w") as f:
        json.dump({
            "compiled": result.compiled,
            "runnable": result.runnable,
            "correct": result.correct,
            "metadata": result.metadata,
        }, f)

def profile_with_hard_timeout_sync(program_path: str, problem_path: str, profile_mode: str, timeout_sec: int, dims: dict | None = None,
                                   machine_config_path: str | None = None, machine_config_preset: str = "default") -> dict:
    fd, result_path = tempfile.mkstemp(prefix="step_profile_", suffix=".json"); os.close(fd)
    p = mp.Process(target=_profile_worker, args=(program_path, problem_path, profile_mode, result_path, dims,
                                                  machine_config_path, machine_config_preset), daemon=True)
    p.start(); p.join(timeout_sec)
    try:
        if p.is_alive():
            p.terminate(); p.join(5)
            return {"compiled": False, "runnable": False, "correct": False,
                    "metadata": {"compilation_error": f"Hard timeout after {timeout_sec}s"}}
        if p.exitcode != 0:
            return {"compiled": False, "runnable": False, "correct": False,
                    "metadata": {"compilation_error": f"Worker process crashed with exit code {p.exitcode}"}}
        with open(result_path) as f:
            return json.load(f)
    except Exception:
        return {"compiled": False, "runnable": False, "correct": False,
                "metadata": {"compilation_error": traceback.format_exc()}}
    finally:
        with contextlib.suppress(Exception): os.remove(result_path)

# -------------------------- Fix-up Helpers --------------------------
def format_errors_for_fixup(metadata: dict, code: str | None = None, preamble_line_count: int = 0) -> str:
    """Format constraint violations and build errors into a readable list for the fix-up prompt.

    *preamble_line_count* is the number of lines (imports, dims, baseline) that precede
    the body in the temp file executed by the profiler. Traceback line numbers from the
    temp file are adjusted by this offset so they map to lines in *code*.
    """
    lines = []
    if "build_error" in metadata:
        tb = metadata.get("build_error_traceback", "")
        line_ref = _extract_code_line_ref(tb, code, preamble_line_count) if tb and code else ""
        lines.append(f"- Build error{line_ref}: {metadata['build_error']}")
        if tb:
            lines.append(f"  Traceback (generated code only):\n{_filter_traceback_to_code(tb, code, preamble_line_count)}")
    for v in metadata.get("constraint_violations", []):
        lines.append(f"- [{v['check']}] {v['message']}")
    return "\n".join(lines) if lines else ""


def _extract_code_line_ref(tb: str, code: str | None, preamble_line_count: int = 0) -> str:
    """Parse traceback to find the last frame inside the generated code and return ' (line N)'."""
    if not code:
        return ""
    import re
    matches = re.findall(r'File ".*?", line (\d+)', tb)
    if not matches:
        return ""
    code_lines = code.splitlines()
    for line_no_str in reversed(matches):
        body_line = int(line_no_str) - preamble_line_count
        if 1 <= body_line <= len(code_lines):
            return f" (line {body_line}: `{code_lines[body_line - 1].strip()}`)"
    return ""


def _filter_traceback_to_code(tb: str, code: str | None, preamble_line_count: int = 0) -> str:
    """Extract traceback frames that reference lines within the generated code's line range."""
    if not code:
        return "  " + tb.strip().splitlines()[-1] if tb.strip() else ""
    import re
    code_lines = code.splitlines()
    code_line_count = len(code_lines)
    result = []
    for match in re.finditer(r'(  File ".*?", line (\d+).*\n(?:    .+\n)?)', tb):
        body_line = int(match.group(2)) - preamble_line_count
        if 1 <= body_line <= code_line_count:
            result.append(f"    line {body_line}: {code_lines[body_line - 1].strip()}")
    # Always include the final exception line
    tb_lines = tb.strip().splitlines()
    if tb_lines:
        result.append(f"    {tb_lines[-1].strip()}")
    return "\n".join(result) if result else ""


def _compute_preamble_line_count(baseline_code: str, values_json: str | None) -> int:
    """Count how many lines precede the body code in the temp file built by _write_temp_kernel."""
    count = STEP_IMPORTS.count("\n") + 1  # imports + trailing newline
    if values_json:
        dims = json.loads(values_json)
        count += len(dims) + 1  # one line per dim + blank line
    count += baseline_code.count("\n") + 1  # baseline + trailing newline
    return count


def _fixup_sync(original_code, errors, system_prompt, fixup_template_path, model_config_path,
                prompts_dir=None, proposal_name=None):
    """Run fix-up LLM call in a subprocess. Returns fixed code string, or None on failure."""
    fd, result_path = tempfile.mkstemp(prefix="fixup_", suffix=".json"); os.close(fd)
    fd2, prompt_path = tempfile.mkstemp(prefix="fixup_prompt_", suffix=".txt"); os.close(fd2)
    with open(fixup_template_path, "r") as f:
        template = f.read()
    # Add line numbers to help the LLM locate errors
    numbered_code = "\n".join(
        f"{i+1:4d} | {line}" for i, line in enumerate(original_code.splitlines())
    )
    operator_ref_path = os.path.join(os.path.dirname(fixup_template_path), "operator_reference.txt")
    with open(operator_ref_path, "r") as f:
        operator_reference = f.read()
    user_prompt = template.replace("{generated_code}", numbered_code).replace("{error_list}", errors).replace("{operator_reference}", operator_reference)
    # Log fixup prompts to checkpoint directory
    if prompts_dir is not None:
        fixup_dir = prompts_dir.parent / "fixup_prompts"
        fixup_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{proposal_name}" if proposal_name else ""
        (fixup_dir / f"system_prompt{suffix}.txt").write_text(system_prompt)
        (fixup_dir / f"user_prompt{suffix}.txt").write_text(user_prompt)
        with open(fixup_template_path, "r") as f:
            (fixup_dir / "fixup_template.txt").write_text(f.read())
    with open(prompt_path, "w") as f:
        json.dump({"system_prompt": system_prompt, "user_prompt": user_prompt}, f)
    import subprocess as sp
    script = f"""
import json, sys, asyncio
sys.path.insert(0, '.')
from src.utils import extract_first_code
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from src.utils import retry_runner_safer

with open('{prompt_path}') as f:
    data = json.load(f)
with open('{model_config_path}') as f:
    mc = json.load(f)

async def run():
    client = AsyncOpenAI(base_url=mc['url'], api_key=mc['api_key'], timeout=60000)
    model = OpenAIChatCompletionsModel(model=mc['model'], openai_client=client)
    set_tracing_disabled(disabled=True)
    agent = Agent(name="Fixup", instructions=data['system_prompt'], model=model)
    result = await retry_runner_safer(agent, data['user_prompt'], max_retries=3, delay=5)
    if result is None:
        return None
    return extract_first_code(result.final_output, ["python"])

code = asyncio.run(run())
with open('{result_path}', 'w') as f:
    json.dump({{"code": code}}, f)
"""
    sp.run([sys.executable, "-c", script], check=False, timeout=120, cwd=os.environ.get("ACCELOPT_BASE_DIR", "."))
    with open(result_path) as f:
        result = json.load(f)
    code = result.get("code")
    with contextlib.suppress(Exception): os.remove(result_path)
    with contextlib.suppress(Exception): os.remove(prompt_path)
    return code

# ---------- Profile and collect results ----------
def profile_and_collect(
    proposals: list[dict],
    baseline_props: StepKernelProperties,
    case_config: ExecutorConfig,
    base_spec: dict,
    per_profile_timeout: int = 900,
    machine_config_path: str | None = None,
    machine_config_preset: str = "default",
    rel_tol: float = 2e-5,
    fixup_template_path: str | None = None,
    model_config_path: str | None = None,
    prompts_dir: Path | None = None,
):
    results = []
    for prop in proposals:
        name, result, code = prop["name"], prop["result"], prop["code"]

        spec = {
            "problem": base_spec["problem"],
            "values": base_spec["values"],
            "case_name": base_spec["case_name"],
            "spec_code": base_spec["spec_code"],
            "baseline_code": base_spec["baseline_code"],
            "plan": case_config.optimization_plan,
            "new_kernel_code": code,
            "baseline_metadata": json.dumps(baseline_props.metadata or {}),
        }

        temp_path = None
        start = time.monotonic()
        print(f"[Profile] START name={name} case={base_spec['case_name']} timeout={per_profile_timeout}s")
        temp_path = _write_temp_kernel(code, base_spec["baseline_code"],
                                      values_json=base_spec.get("values"))
        dims = json.loads(base_spec["values"]) if base_spec.get("values") else None
        kp = profile_with_hard_timeout_sync(
            program_path=temp_path,
            problem_path=base_spec.get("problem_path", ""),
            profile_mode=case_config.profile_mode,
            timeout_sec=per_profile_timeout,
            dims=dims,
            machine_config_path=machine_config_path,
            machine_config_preset=machine_config_preset,
        )

        # --- Fix-up loop (retries until no build/syntax error) ---
        MAX_FIXUP_ATTEMPTS = 3
        original_code = code
        fixup_attempted = False
        preamble_lc = _compute_preamble_line_count(
            base_spec["baseline_code"], base_spec.get("values"),
        )
        if fixup_template_path and model_config_path:
            for fixup_round in range(MAX_FIXUP_ATTEMPTS):
                metadata = kp.get("metadata", {})
                has_build_error = metadata.get("build_error")
                has_constraint = metadata.get("constraint_violations")
                if kp.get("correct", False) or not (has_build_error or has_constraint):
                    break
                fixup_attempted = True
                error_desc = format_errors_for_fixup(metadata, code=code, preamble_line_count=preamble_lc)
                logger.info("[Fixup] Attempt %d/%d for %s: %s",
                            fixup_round + 1, MAX_FIXUP_ATTEMPTS, name, error_desc[:200])
                fixed_code = _fixup_sync(
                    code, error_desc, case_config.system_prompt,
                    fixup_template_path, model_config_path,
                    prompts_dir=prompts_dir,
                    proposal_name=f"{name}_round{fixup_round}",
                )
                if not fixed_code:
                    logger.info("[Fixup] LLM returned no code on attempt %d for %s", fixup_round + 1, name)
                    break
                temp_path_fixed = _write_temp_kernel(
                    fixed_code, base_spec["baseline_code"],
                    values_json=base_spec.get("values"),
                )
                kp = profile_with_hard_timeout_sync(
                    program_path=temp_path_fixed,
                    problem_path=base_spec.get("problem_path", ""),
                    profile_mode=case_config.profile_mode,
                    timeout_sec=per_profile_timeout,
                    dims=dims,
                    machine_config_path=machine_config_path,
                    machine_config_preset=machine_config_preset,
                )
                with contextlib.suppress(Exception): os.remove(temp_path_fixed)
                # Always adopt the fixed code — it's the latest attempt
                code = fixed_code
                if kp.get("correct", False):
                    logger.info("[Fixup] Fix-up succeeded on attempt %d for %s", fixup_round + 1, name)
                    break
                logger.info("[Fixup] Attempt %d still has errors for %s", fixup_round + 1, name)

        record_result = {
            "body": code,
            "spec_code": spec["spec_code"],
            "baseline": spec["baseline_code"],
            "problem": spec["problem"],
            "values": spec["values"],
            "kernel_metadata": json.dumps(kp.get("metadata", {})),
            "baseline_metadata": spec["baseline_metadata"],
        }
        if fixup_attempted:
            record_result["original_body"] = original_code

        # Check if there were errors
        if not kp.get("compiled", False) or not kp.get("runnable", False) or not kp.get("correct", False):
            metadata = kp.get("metadata", {})
            error_msg = (metadata.get("compilation_error") or
                       metadata.get("correctness_error") or
                       metadata.get("run_error") or
                       metadata.get("build_error") or
                       "Unknown error")
            record_result["error"] = error_msg
        else:
            # Success — add STeP metrics
            metadata = kp.get("metadata", {})
            record_result["off_chip_bytes"] = metadata.get("off_chip_bytes")
            record_result["on_chip_bytes"] = metadata.get("on_chip_bytes")
            record_result["cycles"] = metadata.get("cycles")
            record_result["off_chip_expr"] = metadata.get("off_chip_expr")
            record_result["on_chip_expr"] = metadata.get("on_chip_expr")
            # Compute speedup (baseline_cycles / candidate_cycles)
            baseline_cycles = (baseline_props.metadata or {}).get("cycles")
            candidate_cycles = metadata.get("cycles")
            if baseline_cycles and candidate_cycles:
                record_result["speedup"] = baseline_cycles / candidate_cycles
            else:
                record_result["speedup"] = None

        elapsed = time.monotonic() - start
        print(f"[Profile] END name={name} case={base_spec['case_name']} elapsed={elapsed}s")
        results.append(record_result)
        if record_result.get("error", None) and "Hard timeout" in record_result["error"]:
            print(f"[Profile] BREAK name={name} case={base_spec['case_name']} elapsed={elapsed}s")
            break
        if temp_path:
            with contextlib.suppress(Exception):
                os.remove(temp_path)

    return results

# ---------- main(): orchestrates proposal generation and profiling ----------
async def process_single_service_plan(
    case_config: ExecutorConfig,
    baseline_props: StepKernelProperties,
    model: OpenAIChatCompletionsModel,
    base_spec: dict,
    machine_config_path: str | None = None,
    machine_config_preset: str = "default",
    rel_tol: float = 2e-5,
    per_profile_timeout: int = 600,
    prompts_dir: Path | None = None,
    fixup_template_path: str | None = None,
    model_config_path: str | None = None,
    executor_experience_path: str | None = None,
):
    # Load executor experiences (successes + failures) if available
    executor_experiences_block = ""
    if executor_experience_path and os.path.exists(executor_experience_path):
        from src.agents.construct_executor_experience import render_executor_experiences, render_executor_failures
        with open(executor_experience_path, "r") as f:
            experiences = json.load(f)
        executor_experiences_block = render_executor_experiences(experiences)
        failures_path = executor_experience_path.replace(".json", "_failures.json")
        if os.path.exists(failures_path):
            with open(failures_path, "r") as f:
                failures = json.load(f)
            failures_block = render_executor_failures(failures)
            if failures_block:
                executor_experiences_block = executor_experiences_block + "\n\n" + failures_block if executor_experiences_block else failures_block

    pconfig = ExecutorPromptConfig(
        host_problem_path=case_config.task_path,
        step_kernel_path=case_config.kernel_path,
        user_template_path=case_config.user_template_path,
        optimization_plan=case_config.optimization_plan,
        include_baseline=case_config.include_baseline,
        values_json=case_config.values,
    )
    if prompts_dir is not None:
        user_prompt = construct_executor_prompt(pconfig, executor_experiences_block=executor_experiences_block)
        (prompts_dir / f"user_prompt_{case_config.service_name}.txt").write_text(user_prompt)
        if executor_experiences_block:
            (prompts_dir / f"executor_experiences_{case_config.service_name}.txt").write_text(executor_experiences_block)
    agent = Agent(name="Executor", instructions=case_config.system_prompt, model=model)

    # 1) LLM parallel proposal generation
    proposals = await stage1_gather_proposals(case_config.service_name, pconfig, agent, case_config.num_samples, executor_experiences_block=executor_experiences_block)

    # 2) Sequential profiling with result collection
    results = profile_and_collect(proposals, baseline_props, case_config, base_spec, per_profile_timeout=per_profile_timeout,
                                        machine_config_path=machine_config_path, machine_config_preset=machine_config_preset,
                                        rel_tol=rel_tol,
                                        fixup_template_path=fixup_template_path, model_config_path=model_config_path,
                                        prompts_dir=prompts_dir)

    return results

# -------------------------- Driver --------------------------
async def main(args):
    os.makedirs(args.exp_dir, exist_ok=True)
    start_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    from src.step_kernel_wrapper import load_machine_config, apply_prompt_substitutions
    mc = load_machine_config(path=args.machine_config_path, preset=args.machine_config_preset)

    with open(args.base_prompt_path, "r") as f:
        system_prompt = f.read()
    # Append operator reference if it exists alongside the base prompt
    operator_ref_path = os.path.join(os.path.dirname(args.base_prompt_path), "operator_reference.txt")
    if os.path.exists(operator_ref_path):
        with open(operator_ref_path, "r") as f:
            system_prompt += "\n\n" + f.read()
    if mc is not None:
        system_prompt = apply_prompt_substitutions(system_prompt, mc)
    # Compute fix-up template path
    fixup_template_path = os.path.join(os.path.dirname(args.base_prompt_path), "fixup_prompt_template.txt")
    if not os.path.exists(fixup_template_path):
        fixup_template_path = None
    # Save the fully assembled executor prompts for reproducibility
    executor_prompts_dir = Path(args.exp_dir) / "executor_prompts"
    executor_prompts_dir.mkdir(parents=True, exist_ok=True)
    (executor_prompts_dir / "system_prompt.txt").write_text(system_prompt)
    with open(args.user_template_path, "r") as f:
        (executor_prompts_dir / "user_prompt_template.txt").write_text(f.read())

    with open(args.problems_path, "r") as f:
        problems = pd.read_csv(f)
    with open(args.extractor_output_path, "r") as f:
        extractor_output_list = json.load(f)
    with open(args.model_config_path, "r") as f:
        model_config = json.load(f)

    # model client
    BASE_URL = model_config['url']
    API_KEY = model_config['api_key']
    LLM_TIMEOUT = 60000
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=LLM_TIMEOUT)
    model = OpenAIChatCompletionsModel(model=model_config["model"], openai_client=client)
    set_tracing_disabled(disabled=True)

    mp.set_start_method("spawn", force=True)

    # Collect all results
    output_list = []

    for _, row in problems.iterrows():
        service_name = row["service_name"]
        logfire.configure(service_name=service_name)
        logfire.instrument_openai()
        plans = next(item["plans"] for item in extractor_output_list if item["service_name"] == service_name)

        # Load baseline profile metadata
        baseline_props = StepKernelProperties()
        baseline_props.metadata = json.loads(row["profile"])

        with open(row["task"], "r") as f:
            spec_code = f.read()
        with open(row["kernel"], "r") as f:
            baseline_code = f.read()

        base_spec = {
            "problem": row["problem"],
            "values": row["values"],
            "case_name": row["case_name"],
            "spec_code": spec_code,
            "baseline_code": baseline_code,
            "problem_path": row["task"],
        }

        single_dict = {"service_name": service_name, "case_name": row["case_name"]}

        for i in range(len(plans)):
            plan = plans[i]
            case_config = ExecutorConfig(
                system_prompt=system_prompt,
                service_name=f"{service_name}_plan_{i}",
                kernel_path=row["kernel"],
                task_path=row["task"],
                optimization_plan=plan,
                problem=row["problem"],
                values=row["values"],
                case_name=row["case_name"],
                num_samples=args.num_samples,
                user_template_path=args.user_template_path,
                profile_mode=args.profile_mode,
                include_baseline=args.include_baseline,
            )

            plan_results = await asyncio.wait_for(
                process_single_service_plan(case_config, baseline_props, model, base_spec,
                                           machine_config_path=args.machine_config_path,
                                           machine_config_preset=args.machine_config_preset,
                                           rel_tol=args.rel_tol,
                                           per_profile_timeout=args.per_profile_timeout,
                                           prompts_dir=executor_prompts_dir,
                                           fixup_template_path=fixup_template_path,
                                           model_config_path=args.model_config_path,
                                           executor_experience_path=getattr(args, 'executor_experience_path', None)),
                timeout=7200
            )

            # Store results for each sample in this plan
            for j in range(args.num_samples):
                if j < len(plan_results):
                    single_dict[f"plan_{i}_{j}"] = plan_results[j]
                else:
                    single_dict[f"plan_{i}_{j}"] = {"error": "No implementation found"}

        output_list.append(single_dict)

    # time record (end)
    end_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # Save results in format similar to read_executor_log output
    output_dict = {
        "exp_date": args.exp_date,
        "executor_dir": args.exp_dir,
        "read_token": args.read_token,
        "executor_start_timestamp": start_time,
        "executor_end_timestamp": end_time,
        "executor_results": output_list
    }

    with open(args.output_path, "w") as f:
        json.dump(output_dict, f, indent=4)

    # Materialize code fields as readable .py files
    materialize_executor_results(output_dict, Path(args.exp_dir))

    # Save timing info
    time_record_path = f"{args.exp_dir}/executor_start_end_time.txt"
    with open(time_record_path, "w") as f:
        f.write(f"{start_time},{end_time}")

    logger.info(f"Results saved to {args.output_path}")

CODE_FIELDS = ("body", "original_body", "baseline", "spec_code")

def materialize_executor_results(output_dict: dict, exp_dir: Path) -> None:
    """Write code fields from executor results as standalone .py files for readability."""
    results_dir = exp_dir / "executor_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for record in output_dict["executor_results"]:
        service_name = record["service_name"]
        for key, value in record.items():
            if key in ("service_name", "case_name") or not isinstance(value, dict):
                continue
            plan_dir = results_dir / f"{service_name}_{key}"
            plan_dir.mkdir(parents=True, exist_ok=True)
            for field in CODE_FIELDS:
                code = value.get(field)
                if code:
                    (plan_dir / f"{field}.py").write_text(code)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--problems_path", type=str, required=True)
    parser.add_argument("--profile_mode", type=str, default="cycle_accurate", choices=["symbolic", "cycle_accurate"])
    parser.add_argument("--extractor_output_path", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--base_prompt_path", type=str, required=True)
    parser.add_argument("--user_template_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--exp_date", type=str, required=True)
    parser.add_argument("--read_token", type=str, default="unused")
    parser.add_argument("--machine_config_path", type=str, default=None)
    parser.add_argument("--machine_config_preset", type=str, default="default")
    parser.add_argument("--rel_tol", type=float, default=2e-5)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--per_profile_timeout", type=int, default=600)
    parser.add_argument("--include_baseline", action="store_true")
    parser.add_argument("--executor_experience_path", type=str, default=None)
    args = parser.parse_args()

    if args.log_file:
        from pathlib import Path
        _root = logging.getLogger()
        for _h in _root.handlers[:]:
            if isinstance(_h, logging.FileHandler):
                _h.close()
                _root.removeHandler(_h)
        _handler = logging.FileHandler(Path(args.log_file))
        _handler.setLevel(logging.INFO)
        _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
        _root.addHandler(_handler)
        _root.setLevel(logging.INFO)

    asyncio.run(main(args))
