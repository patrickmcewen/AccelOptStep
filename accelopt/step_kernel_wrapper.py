"""STeP kernel profiler: symbolic analysis + cycle-accurate simulation with correctness."""

import importlib.util
import os
import sys
import tempfile
from enum import Enum

import torch
import sympy
from networkx import MultiDiGraph

from accelopt.eval_step import StepKernelProperties

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
        hbm_config: dict | None = None,
        sim_config: dict | None = None,
    ):
        self.step_code = step_code
        self.problem_path = problem_path
        self.profile_mode = profile_mode
        self.hbm_config = hbm_config or {
            "addr_offset": 64,
            "channel_num": 32,
            "per_channel_latency": 2,
            "per_channel_init_interval": 2,
            "per_channel_outstanding": 1,
            "per_channel_start_up_time": 14,
        }
        self.sim_config = sim_config or {
            "channel_depth": 2,
            "functional_sim": True,
            "mock_bf16": False,
        }

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

        if not hasattr(mod, "compute_gold"):
            os.chdir(orig_dir)
            result["correct"] = False
            result["correctness_error"] = "Generated code missing compute_gold()"
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
        gold = mod.compute_gold().float()

        if sim_tensor.shape != gold.shape:
            result["correct"] = False
            result["correctness_error"] = (
                f"Shape mismatch: sim={tuple(sim_tensor.shape)} gold={tuple(gold.shape)}"
            )
            return result

        max_diff = (sim_tensor - gold).abs().max().item()
        passed = torch.allclose(sim_tensor, gold, rtol=RTOL, atol=ATOL)

        result["correct"] = passed
        result["max_diff"] = max_diff
        if not passed:
            result["correctness_error"] = f"Max abs diff: {max_diff}"

        return result
