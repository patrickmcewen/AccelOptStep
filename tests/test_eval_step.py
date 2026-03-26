import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from accelopt.eval_step import check_step_correctness, StepKernelProperties


def test_correct_output_passes():
    gold = torch.randn(32, 64)
    sim_output = gold.clone()
    result = check_step_correctness(sim_output, gold, rtol=1e-4, atol=1e-6)
    assert result.correct is True
    assert result.runnable is True


def test_incorrect_output_fails():
    gold = torch.randn(32, 64)
    sim_output = torch.randn(32, 64)
    result = check_step_correctness(sim_output, gold, rtol=1e-4, atol=1e-6)
    assert result.correct is False
    assert result.runnable is True


def test_shape_mismatch_fails():
    gold = torch.randn(32, 64)
    sim_output = torch.randn(32, 32)
    result = check_step_correctness(sim_output, gold, rtol=1e-4, atol=1e-6)
    assert result.correct is False
    assert result.runnable is False
    assert "Shape mismatch" in result.metadata["error"]


def test_close_output_passes():
    gold = torch.ones(10, 10)
    sim_output = gold + 1e-7  # very small diff
    result = check_step_correctness(sim_output, gold, rtol=1e-4, atol=1e-6)
    assert result.correct is True


def test_properties_default():
    props = StepKernelProperties()
    assert props.compiled is False
    assert props.correct is False
    assert props.runnable is False
    assert props.metadata == {}
