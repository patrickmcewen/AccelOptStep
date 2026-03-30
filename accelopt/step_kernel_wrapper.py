"""STeP kernel profiler: symbolic analysis + cycle-accurate simulation with correctness."""

import importlib.util
import os
import sys
import tempfile
from enum import Enum

import yaml
import torch
import sympy
from networkx import MultiDiGraph

from accelopt.eval_step import StepKernelProperties

# Default machine config path (relative to this file -> StepBench/machine_config.yaml)
_DEFAULT_MACHINE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "StepBench", "machine_config.yaml"
)


def load_machine_config(path: str | None = None, preset: str = "default") -> dict:
    """Load machine config from YAML, selecting a named preset.

    Returns dict with keys: total_compute_bw, hbm, sim.
    """
    path = path or _DEFAULT_MACHINE_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f)
    assert preset in raw, (
        f"Machine config preset '{preset}' not found in {path}. "
        f"Available: {list(raw.keys())}"
    )
    return raw[preset]


def prompt_substitutions(mc: dict) -> dict[str, str]:
    """Return a dict of {placeholder: value} for prompt template substitution."""
    hbm = mc["hbm"]
    on_chip_mb = mc["on_chip_memory_bytes"] / (1024 * 1024)
    on_chip_human = f"{int(on_chip_mb)} MB" if on_chip_mb == int(on_chip_mb) else f"{on_chip_mb:.1f} MB"
    return {
        "{total_compute_bw}": str(mc["total_compute_bw"]),
        "{on_chip_memory_bytes}": str(mc["on_chip_memory_bytes"]),
        "{on_chip_memory_human}": on_chip_human,
        "{hbm_channel_num}": str(hbm["channel_num"]),
        "{hbm_per_channel_latency}": str(hbm["per_channel_latency"]),
        "{hbm_per_channel_init_interval}": str(hbm["per_channel_init_interval"]),
        "{hbm_per_channel_outstanding}": str(hbm["per_channel_outstanding"]),
        "{hbm_per_channel_start_up_time}": str(hbm["per_channel_start_up_time"]),
        "{hbm_addr_offset}": str(hbm["addr_offset"]),
    }


def apply_prompt_substitutions(text: str, mc: dict) -> str:
    """Replace all machine config placeholders in a prompt string."""
    for placeholder, value in prompt_substitutions(mc).items():
        text = text.replace(placeholder, value)
    return text


# Ensure step_artifact is importable
STEP_SRC = os.environ.get(
    "STEP_ARTIFACT_SRC",
    os.path.join(os.path.dirname(__file__), "../../step_artifact/src"),
)
if STEP_SRC not in sys.path:
    sys.path.insert(0, STEP_SRC)

RTOL = 1e-3
ATOL = 1e-3


# Kept for backward compatibility in callers that reference ProfileMode.
class ProfileMode(Enum):
    SYMBOLIC = "symbolic"
    CYCLE_ACCURATE = "cycle_accurate"


class StepKernel:
    def __init__(
        self,
        step_code: str,
        problem_path: str,
        profile_mode: ProfileMode = ProfileMode.CYCLE_ACCURATE,
        dims: dict | None = None,
        machine_config_path: str | None = None,
        machine_config_preset: str = "default",
    ):
        self.step_code = step_code
        self.problem_path = problem_path
        self.profile_mode = profile_mode
        self.dims = dims

        mc = load_machine_config(path=machine_config_path, preset=machine_config_preset)
        self.total_compute_bw = mc["total_compute_bw"]
        self.on_chip_memory_bytes = mc["on_chip_memory_bytes"]
        self.hbm_config = mc["hbm"]
        self.sim_config = mc["sim"]

    def profile(self) -> StepKernelProperties:
        """Build graph, run symbolic analysis, and (if cycle_accurate) run sim + correctness.

        SYMBOLIC mode:  symbolic profiling only (fast, no Docker needed).
        CYCLE_ACCURATE: symbolic profiling + cycle-accurate simulation + correctness check.
        """
        props = StepKernelProperties()

        # Step 1: Execute the generated code to get the graph
        mod, graph, output_op = self._execute_step_code()
        assert graph is not None, "build_graph() returned None for graph"
        assert output_op is not None, "build_graph() returned None for output_op"
        props.compiled = True
        props.runnable = True

        # Step 2: Symbolic profiling (always — gives memory traffic estimates for LLM feedback)
        symbolic_metrics = self._symbolic_profile(graph)
        props.metadata.update(symbolic_metrics)

        # Step 2.5: Validate hardware constraints
        total_used = sum(node.compute_bw for node in graph.nodes() if hasattr(node, "compute_bw"))
        props.metadata["total_compute_bw_used"] = total_used
        if total_used > self.total_compute_bw:
            props.correct = False
            props.metadata["correctness_error"] = (
                f"Compute bandwidth budget exceeded: used {total_used}, limit {self.total_compute_bw}"
            )
            return props

        on_chip_bytes = symbolic_metrics.get("on_chip_bytes")
        if isinstance(on_chip_bytes, (int, float)) and on_chip_bytes > self.on_chip_memory_bytes:
            props.correct = False
            props.metadata["correctness_error"] = (
                f"On-chip memory exceeded: {int(on_chip_bytes)} bytes, limit {self.on_chip_memory_bytes} bytes"
            )
            return props

        # Step 3: Cycle-accurate simulation + correctness check
        if self.profile_mode == ProfileMode.CYCLE_ACCURATE:
            sim_metrics = self._cycle_accurate_profile(graph, output_op, mod)
            props.metadata.update(sim_metrics)
            props.correct = sim_metrics.get("correct", False)
            if not props.correct:
                props.metadata["correctness_error"] = sim_metrics.get(
                    "correctness_error", "Unknown correctness failure"
                )
        else:
            # Symbolic-only: no correctness check possible
            props.correct = True

        return props

    def _execute_step_code(self) -> tuple[object, MultiDiGraph, object]:
        """Write code to temp file, import it, call build_graph().

        Returns (module, graph, output_op). The module is kept alive so
        compute_gold() can be called later for correctness checking.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(self.step_code)
            tmp_path = f.name

        spec = importlib.util.spec_from_file_location("step_kernel", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        os.unlink(tmp_path)

        assert hasattr(mod, "build_graph"), "Generated code must define build_graph()"
        # LLM-generated code may define build_graph() or build_graph(dims).
        # Inspect the signature and call accordingly.
        import inspect
        sig = inspect.signature(mod.build_graph)
        if len(sig.parameters) > 0 and self.dims is not None:
            graph, output_op = mod.build_graph(self.dims)
        else:
            graph, output_op = mod.build_graph()
        return mod, graph, output_op

    def _symbolic_profile(self, graph: MultiDiGraph) -> dict:
        """Traverse graph nodes, sum off_chip_traffic and on_chip_requirement."""
        from step_py.ops import StepOps

        total_off_chip = sympy.Integer(0)
        total_on_chip = sympy.Integer(0)

        for node in graph.nodes():
            if isinstance(node, StepOps):
                total_off_chip = sympy.Add(total_off_chip, node.off_chip_traffic())
                total_on_chip = sympy.Add(total_on_chip, node.on_chip_requirement())

        off_chip_val = float(total_off_chip) if total_off_chip.is_number else str(total_off_chip)
        on_chip_val = float(total_on_chip) if total_on_chip.is_number else str(total_on_chip)

        return {
            "off_chip_bytes": off_chip_val,
            "on_chip_bytes": on_chip_val,
            "off_chip_expr": str(total_off_chip),
            "on_chip_expr": str(total_on_chip),
        }

    def _cycle_accurate_profile(self, graph: MultiDiGraph, output_op, mod) -> dict:
        """Run cycle-accurate simulator and check correctness against compute_gold()."""
        from sim import simulate, HBMConfig, SimConfig
        from utils.gold_checking import reconstruct_numpy

        hbm = HBMConfig(**self.hbm_config)
        sim = SimConfig(
            channel_depth=self.sim_config.get("channel_depth"),
            functional_sim=self.sim_config.get("functional_sim", True),
            mock_bf16=self.sim_config.get("mock_bf16", False),
        )

        # Simulator writes .npy output files to CWD, so use a temp directory.
        orig_dir = os.getcwd()
        tmpdir = tempfile.mkdtemp(prefix="step_sim_")
        pb_path = os.path.join(tmpdir, "graph.pb")

        os.chdir(tmpdir)
        cycles, duration_ms, duration_s = simulate(
            graph,
            logging=False,
            hbm_config=hbm,
            sim_config=sim,
            protobuf_file=pb_path,
            db_name=None,
        )

        # Correctness: compare simulator output to compute_gold()
        result = {
            "cycles": cycles,
            "duration_ms": duration_ms,
            "duration_s": duration_s,
        }

        # Get compute_gold: prefer problem file (new API), fall back to baseline (old API)
        gold_source = mod
        gold_args = ()
        if self.dims is not None and self.problem_path:
            problem_spec = importlib.util.spec_from_file_location("problem_mod", self.problem_path)
            problem_mod = importlib.util.module_from_spec(problem_spec)
            problem_spec.loader.exec_module(problem_mod)
            if hasattr(problem_mod, "compute_gold"):
                gold_source = problem_mod
                gold_args = (self.dims,)

        if not hasattr(gold_source, "compute_gold"):
            os.chdir(orig_dir)
            result["correct"] = False
            result["correctness_error"] = "No compute_gold() found in problem or baseline"
            return result

        store_name = output_op.store_file_name
        if not os.path.exists(f"{store_name}.json") or not os.path.exists(f"{store_name}.npy"):
            os.chdir(orig_dir)
            result["correct"] = False
            result["correctness_error"] = "Simulation did not produce output files (possible Rust panic or store failure)"
            return result
        sim_output = reconstruct_numpy(store_name, delete_npy=False)
        os.chdir(orig_dir)

        sim_tensor = torch.from_numpy(sim_output).float()
        gold = gold_source.compute_gold(*gold_args).float()

        # Baselines may flatten batch dimensions (e.g. SDPA merges batch*heads),
        # so compare by total element count after flattening both.
        if sim_tensor.numel() != gold.numel():
            result["correct"] = False
            result["correctness_error"] = (
                f"Element count mismatch: sim={sim_tensor.numel()} gold={gold.numel()} "
                f"(sim shape={tuple(sim_tensor.shape)}, gold shape={tuple(gold.shape)})"
            )
            return result

        sim_flat = sim_tensor.reshape(-1)
        gold_flat = gold.reshape(-1)
        max_diff = (sim_flat - gold_flat).abs().max().item()
        passed = torch.allclose(sim_flat, gold_flat, rtol=RTOL, atol=ATOL)

        result["correct"] = passed
        result["max_diff"] = max_diff
        if not passed:
            result["correctness_error"] = f"Max abs diff: {max_diff}"

        return result
