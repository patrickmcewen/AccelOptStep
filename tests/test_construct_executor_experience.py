import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.construct_executor_experience import select_executor_experiences, render_executor_experiences


def _make_results(plans_per_entry):
    """Helper: plans_per_entry is list of (case_name, [(body, speedup, error_or_None), ...])"""
    entries = []
    for case_name, plans in plans_per_entry:
        entry = {"service_name": "svc", "case_name": case_name}
        for i, (body, speedup, error) in enumerate(plans):
            d = {"body": body, "speedup": speedup}
            if error is not None:
                d["error"] = error
            entry[f"plan_0_{i}"] = d
        entries.append(entry)
    return {"executor_results": entries}


def test_selects_correct_fast_shortest():
    short_code = "def build_graph(d):\n  return d"
    long_code = "def build_graph(d):\n  x = d\n  y = x\n  return y"
    results = _make_results([
        ("gemm_256", [
            (short_code, 2.0, None),
            (long_code, 2.5, None),
        ]),
    ])
    selected = select_executor_experiences(results, max_examples=5)
    assert len(selected) == 1
    # Should pick shortest code for the case
    assert selected[0]["code"] == short_code
    assert selected[0]["case_name"] == "gemm_256"


def test_filters_out_errors():
    results = _make_results([
        ("case_a", [
            ("def f(): pass", 3.0, "Shape mismatch"),
        ]),
        ("case_b", [
            ("def g(): pass", 2.0, None),
        ]),
    ])
    selected = select_executor_experiences(results, max_examples=5)
    assert len(selected) == 1
    assert selected[0]["case_name"] == "case_b"


def test_filters_out_slowdowns():
    results = _make_results([
        ("case_slow", [
            ("def f(): pass", 0.8, None),
            ("def g(): pass", 1.0, None),  # equal to baseline, not faster
        ]),
        ("case_fast", [
            ("def h(): pass", 1.5, None),
        ]),
    ])
    selected = select_executor_experiences(results, max_examples=5)
    assert len(selected) == 1
    assert selected[0]["case_name"] == "case_fast"


def test_caps_at_max_examples():
    results = _make_results([
        ("case_a", [("def a(): pass", 3.0, None)]),
        ("case_b", [("def b(): pass", 2.0, None)]),
        ("case_c", [("def c(): pass", 1.5, None)]),
    ])
    selected = select_executor_experiences(results, max_examples=2)
    assert len(selected) == 2
    # Sorted by speedup descending
    assert selected[0]["case_name"] == "case_a"
    assert selected[1]["case_name"] == "case_b"


def test_empty_results():
    assert select_executor_experiences({"executor_results": []}) == []
    assert select_executor_experiences({}) == []


def test_render_empty():
    assert render_executor_experiences([]) == ""


def test_render_nonempty():
    experiences = [
        {"case_name": "gemm_256", "speedup": 2.5, "code": "def build_graph(d):\n  return d"},
    ]
    rendered = render_executor_experiences(experiences)
    assert "# Reference: Verified STeP IR Optimizations" in rendered
    assert "## Example: gemm_256 (2.5x speedup)" in rendered
    assert "```python" in rendered
    assert "def build_graph(d):" in rendered
