# Merged Executor - Direct results collection without logfire
# Adapted for STeP IR (from NKI version)

import os
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
from accelopt.utils import extract_first_code, retry_runner_safer
from accelopt.eval_step import StepKernelProperties
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunConfig, ModelSettings

# -------------------------- Logging --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------- STeP Import Preamble --------------------------
STEP_IMPORTS = """
import torch
import torch.nn as nn
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
"""

# -------------------------- NKI Import Preamble --------------------------
NKI_IMPORTS = """
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim
"""

# -------------------------- Config Models --------------------------
class ExecutorPromptConfig(BaseModel):
    host_problem_path: str = ""
    step_kernel_path: str = ""
    user_template_path: str = ""
    optimization_plan: str = ""

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

# -------------------------- Helpers --------------------------
def construct_executor_prompt(config: ExecutorPromptConfig) -> str:
    with open(config.host_problem_path, "r") as f:
        host_problem_function = f.read()
    with open(config.step_kernel_path, "r") as f:
        step_kernel_function = f.read()
    with open(config.user_template_path, "r") as f:
        prompt_template = f.read()
    user_prompt = (
        prompt_template
        .replace("{problem_code}", host_problem_function)
        .replace("{kernel_code}", step_kernel_function)
        .replace("{optimization_plan}", config.optimization_plan)
    )
    return user_prompt

def _write_temp_kernel(code: str, baseline_code: str, code_preamble: str = "step") -> str:
    fd, temp_path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        if code_preamble == "nki":
            f.write(NKI_IMPORTS)
            f.write("\n")
            f.write(code)
            f.write("\n")
        else:
            f.write(STEP_IMPORTS)
            f.write("\n")
            f.write(baseline_code)
            f.write("\n")
            f.write(code)
            f.write("\n")
    return temp_path

# -------------------------- Parallel LLM --------------------------
async def propose_once(name: str, config: ExecutorPromptConfig, agent: Agent):
    try:
        user_prompt = construct_executor_prompt(config)
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
        if result is None:
            return None
        code = extract_first_code(result.final_output, ["python"])
        if not code:
            return None
        return {"name": name, "result": result, "code": code}
    except asyncio.TimeoutError:
        logger.warning("LLM timed out for %s", name)
        return None
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("propose_once failed for %s", name)
        return None

# ---------- Stage 1: parallel LLM (async) ----------
async def stage1_gather_proposals(service_name: str, pconfig: ExecutorPromptConfig, base_agent: Agent, num_samples: int):
    tasks = []
    for i in range(num_samples):
        # fresh agent per task to avoid cross-cancellation
        agent = Agent(name=f"Executor_{i}", instructions=base_agent.instructions, model=base_agent.model)
        tasks.append(asyncio.create_task(propose_once(f"{service_name}_{i}", pconfig, agent)))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]

def _profile_worker(program_path, problem_path, profile_mode, result_path, dims=None,
                    machine_config_path=None, machine_config_preset="default"):
    """Profile a STeP kernel. Runs in a subprocess for isolation."""
    import json, traceback
    try:
        from accelopt.step_kernel_wrapper import StepKernel, ProfileMode

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
    except Exception:
        with open(result_path, "w") as f:
            json.dump({
                "compiled": False, "runnable": False, "correct": False,
                "metadata": {"compilation_error": traceback.format_exc()},
            }, f)

def _nki_profile_worker(program_path, base_numpy_path, result_path, rel_tol):
    """Profile an NKI kernel. Runs in a subprocess for isolation."""
    import json, traceback
    try:
        from accelopt.nki_kernel_wrapper import NKIKernel
        k = NKIKernel(program_path, base_numpy_path)
        k.rel_tol = rel_tol
        k.profile([])
        out = {"compiled": k.res.compiled, "runnable": k.res.runnable, "correct": k.res.correct, "metadata": k.res.metadata or {}}
    except Exception:
        out = {"compiled": False, "runnable": False, "correct": False,
               "metadata": {"compilation_error": traceback.format_exc()}}
    with open(result_path, "w") as f:
        json.dump(out, f)

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

def nki_profile_with_hard_timeout_sync(program_path: str, base_numpy_path: str, rel_tol: float, timeout_sec: int) -> dict:
    fd, result_path = tempfile.mkstemp(prefix="nki_profile_", suffix=".json"); os.close(fd)
    p = mp.Process(target=_nki_profile_worker, args=(program_path, base_numpy_path, result_path, rel_tol), daemon=True)
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
    finally:
        with contextlib.suppress(Exception): os.remove(result_path)

# ---------- Stage 2: sequential profiling with result collection ----------
def stage2_profile_and_collect(
    proposals: list[dict],
    baseline_props: StepKernelProperties,
    case_config: ExecutorConfig,
    base_spec: dict,
    per_profile_timeout: int = 900,
    machine_config_path: str | None = None,
    machine_config_preset: str = "default",
    profiler: str = "step",
    speedup_metric: str = "cycles",
    code_preamble: str = "step",
    rel_tol: float = 2e-5,
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
        try:
            start = time.monotonic()
            print(f"[Stage2] START name={name} case={base_spec['case_name']} timeout={per_profile_timeout}s")
            temp_path = _write_temp_kernel(code, base_spec["baseline_code"], code_preamble=code_preamble)
            if profiler == "nki":
                kp = nki_profile_with_hard_timeout_sync(
                    program_path=temp_path,
                    base_numpy_path=base_spec.get("problem_path", ""),
                    rel_tol=rel_tol,
                    timeout_sec=per_profile_timeout,
                )
            else:
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

            record_result = {
                "body": code,
                "spec_code": spec["spec_code"],
                "baseline": spec["baseline_code"],
                "problem": spec["problem"],
                "values": spec["values"],
                "kernel_metadata": json.dumps(kp.get("metadata", {})),
                "baseline_metadata": spec["baseline_metadata"],
            }

            # Check if there were errors
            if not kp.get("compiled", False) or not kp.get("runnable", False) or not kp.get("correct", False):
                metadata = kp.get("metadata", {})
                error_msg = (metadata.get("compilation_error") or
                           metadata.get("correctness_error") or
                           metadata.get("run_error") or
                           "Unknown error")
                record_result["error"] = error_msg
            else:
                metadata = kp.get("metadata", {})
                if speedup_metric == "latency":
                    # Success case - add NKI metrics
                    record_result["latency"] = metadata.get("latency")
                    bl = (baseline_props.metadata or {}).get("latency")
                    cl = metadata.get("latency")
                    if bl and cl:
                        record_result["speedup"] = bl / cl
                    else:
                        record_result["speedup"] = None
                else:
                    # Success case - add STeP metrics
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
            print(f"[Stage2] END name={name} case={base_spec['case_name']} elapsed={elapsed}s")
            results.append(record_result)
            if record_result.get("error", None) and "Hard timeout" in record_result["error"]:
                print(f"[Stage2] BREAK name={name} case={base_spec['case_name']} elapsed={elapsed}s")
                break
        except Exception as e:
            logger.error(f"[Profile Error] {name}: {e}")
            error_result = {
                "error": str(e),
                "body": code,
                "spec_code": spec["spec_code"],
                "baseline": spec["baseline_code"],
                "problem": spec["problem"],
                "values": spec["values"]
            }
            results.append(error_result)
        finally:
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
    profiler: str = "step",
    speedup_metric: str = "cycles",
    code_preamble: str = "step",
    rel_tol: float = 2e-5,
):
    pconfig = ExecutorPromptConfig(
        host_problem_path=case_config.task_path,
        step_kernel_path=case_config.kernel_path,
        user_template_path=case_config.user_template_path,
        optimization_plan=case_config.optimization_plan,
    )
    agent = Agent(name="Executor", instructions=case_config.system_prompt, model=model)

    # 1) LLM parallel proposal generation
    proposals = await stage1_gather_proposals(case_config.service_name, pconfig, agent, case_config.num_samples)

    # 2) Sequential profiling with result collection
    results = stage2_profile_and_collect(proposals, baseline_props, case_config, base_spec, per_profile_timeout=180,
                                        machine_config_path=machine_config_path, machine_config_preset=machine_config_preset,
                                        profiler=profiler, speedup_metric=speedup_metric,
                                        code_preamble=code_preamble, rel_tol=rel_tol)

    return results

# -------------------------- Driver --------------------------
async def main(args):
    # time record (start)
    os.makedirs(args.exp_dir, exist_ok=True)
    start_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # load inputs and substitute machine config placeholders
    from pipeline_registry import resolve_pipeline
    pipeline = resolve_pipeline(args.pipeline)

    if pipeline["needs_machine_config"]:
        from accelopt.step_kernel_wrapper import load_machine_config, apply_prompt_substitutions
        mc = load_machine_config(path=args.machine_config_path, preset=args.machine_config_preset)
    else:
        mc = None

    with open(args.base_prompt_path, "r") as f:
        system_prompt = f.read()
    # Append operator reference if it exists alongside the base prompt
    operator_ref_path = os.path.join(os.path.dirname(args.base_prompt_path), "operator_reference.txt")
    if os.path.exists(operator_ref_path):
        with open(operator_ref_path, "r") as f:
            system_prompt += "\n\n" + f.read()
    if mc is not None:
        system_prompt = apply_prompt_substitutions(system_prompt, mc)
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

    # iterate services
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
            )

            plan_results = await asyncio.wait_for(
                process_single_service_plan(case_config, baseline_props, model, base_spec,
                                           machine_config_path=args.machine_config_path,
                                           machine_config_preset=args.machine_config_preset,
                                           profiler=pipeline["profiler"],
                                           speedup_metric=pipeline["speedup_metric"],
                                           code_preamble=pipeline["code_preamble"],
                                           rel_tol=args.rel_tol),
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

CODE_FIELDS = ("body", "baseline", "spec_code")

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
    parser.add_argument("--read_token", type=str, default="unused")  # For compatibility
    parser.add_argument("--machine_config_path", type=str, default=None, help="Path to machine_config.yaml")
    parser.add_argument("--machine_config_preset", type=str, default="default", help="Preset name in machine_config.yaml")
    parser.add_argument("--pipeline", type=str, default="pytorch-step")
    parser.add_argument("--rel_tol", type=float, default=2e-5, help="Relative tolerance for NKI correctness checks")
    parser.add_argument("--log_file", type=str, default=None, help="Path to per-problem debug log file")
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
