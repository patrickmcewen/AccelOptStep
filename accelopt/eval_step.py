"""Correctness checking for STeP simulation outputs."""

from dataclasses import dataclass, field
import torch


@dataclass
class StepKernelProperties:
    compiled: bool = False
    correct: bool = False
    runnable: bool = False
    metadata: dict = field(default_factory=dict)


def check_step_correctness(
    sim_output: torch.Tensor,
    gold: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> StepKernelProperties:
    """Compare STeP simulation output against PyTorch reference."""
    props = StepKernelProperties()
    assert sim_output is not None, "sim_output is None"
    assert gold is not None, "gold is None"

    if sim_output.numel() != gold.numel():
        props.metadata["error"] = (
            f"Element count mismatch: sim={sim_output.numel()} gold={gold.numel()} "
            f"(sim shape={tuple(sim_output.shape)}, gold shape={tuple(gold.shape)})"
        )
        return props

    sim_flat = sim_output.reshape(-1)
    gold_flat = gold.reshape(-1)
    props.runnable = True
    props.correct = torch.allclose(sim_flat, gold_flat, rtol=rtol, atol=atol)
    if not props.correct:
        max_diff = (sim_flat - gold_flat).abs().max().item()
        props.metadata["error"] = f"Max abs diff: {max_diff}"

    return props
