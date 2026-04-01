import json
import logging

logger = logging.getLogger(__name__)


def select_executor_experiences(executor_results: dict, max_examples: int = 2) -> list[dict]:
    """Select best kernel codes from prior executor results as few-shot examples.

    Picks shortest correct kernel (by line count) per case_name, sorted by speedup descending.
    """
    results_list = executor_results.get("executor_results", [])

    # Collect all valid results: no error, speedup > 1.0, has body
    valid = []
    for entry in results_list:
        case_name = entry.get("case_name", "unknown")
        for key, val in entry.items():
            if not key.startswith("plan_"):
                continue
            assert isinstance(val, dict), f"Expected dict for {key}, got {type(val)}"
            if val.get("error"):
                continue
            speedup = val.get("speedup")
            if speedup is None or speedup <= 1.0:
                continue
            body = val.get("body")
            if not body:
                continue
            valid.append({
                "case_name": case_name,
                "speedup": speedup,
                "code": body,
            })

    # Group by case_name, pick shortest code per case
    by_case: dict[str, list[dict]] = {}
    for v in valid:
        by_case.setdefault(v["case_name"], []).append(v)

    best_per_case = []
    for case_name, candidates in by_case.items():
        shortest = min(candidates, key=lambda c: c["code"].count("\n"))
        best_per_case.append(shortest)

    # Sort by speedup descending, cap at max_examples
    best_per_case.sort(key=lambda c: c["speedup"], reverse=True)
    selected = best_per_case[:max_examples]

    logger.info("Selected %d executor experiences from %d valid results across %d cases",
                len(selected), len(valid), len(by_case))
    return selected


def select_executor_failures(executor_results: dict, max_examples: int = 2) -> list[dict]:
    """Select representative failures from prior executor results as negative examples.

    Picks one failure per distinct error type (build_error text or first constraint check),
    preferring shorter code. Returns up to *max_examples* entries.
    """
    results_list = executor_results.get("executor_results", [])

    failures = []
    for entry in results_list:
        case_name = entry.get("case_name", "unknown")
        for key, val in entry.items():
            if not key.startswith("plan_"):
                continue
            assert isinstance(val, dict), f"Expected dict for {key}, got {type(val)}"
            if not val.get("error"):
                continue
            body = val.get("body")
            if not body:
                continue
            # Parse kernel_metadata to get structured error info
            meta = json.loads(val.get("kernel_metadata", "{}"))
            build_err = meta.get("build_error", "")
            violations = meta.get("constraint_violations", [])
            # Derive an error category for deduplication
            if build_err:
                error_key = build_err.split("\n")[0][:120]
                error_desc = build_err
            elif violations:
                error_key = violations[0].get("check", "unknown")
                error_desc = "; ".join(
                    f"[{v['check']}] {v['message']}" for v in violations
                )
            else:
                error_key = val["error"][:120]
                error_desc = val["error"]
            failures.append({
                "case_name": case_name,
                "error_key": error_key,
                "error_desc": error_desc,
                "code": body,
            })

    # Deduplicate: one per error_key, pick shortest code
    by_key: dict[str, list[dict]] = {}
    for f in failures:
        by_key.setdefault(f["error_key"], []).append(f)

    unique = []
    for error_key, candidates in by_key.items():
        shortest = min(candidates, key=lambda c: c["code"].count("\n"))
        unique.append(shortest)

    selected = unique[:max_examples]
    logger.info("Selected %d executor failures from %d total failures across %d error types",
                len(selected), len(failures), len(by_key))
    return selected


def render_executor_failures(failures: list[dict]) -> str:
    """Format failures into a prompt block warning the executor about common mistakes."""
    if not failures:
        return ""

    parts = [
        "# Common Mistakes to Avoid",
        "The following build_graph() attempts from prior iterations failed. "
        "Study the errors so you do not repeat them.",
    ]
    for f in failures:
        parts.append(f"\n## Failed: {f['case_name']}")
        parts.append(f"**Error:** {f['error_desc']}")
        parts.append(f"```python\n{f['code']}\n```")

    return "\n".join(parts)


def render_executor_experiences(experiences: list[dict]) -> str:
    """Format experiences into a prompt block for the executor."""
    if not experiences:
        return ""

    parts = [
        "# Reference: Verified STeP IR Optimizations",
        "The following build_graph() functions were generated in prior iterations, "
        "verified correct, and achieved speedups over baseline. Use them as reference "
        "for STeP IR patterns and idioms.",
    ]
    for exp in experiences:
        parts.append(f"\n## Example: {exp['case_name']} ({exp['speedup']}x speedup)")
        parts.append(f"```python\n{exp['code']}\n```")

    return "\n".join(parts)


def main(args):
    with open(args.executor_results_path, "r") as f:
        executor_results = json.load(f)

    selected = select_executor_experiences(executor_results, max_examples=args.max_examples)
    with open(args.output_path, "w") as f:
        json.dump(selected, f, indent=4)
    logger.info("Wrote %d executor experiences to %s", len(selected), args.output_path)

    failures = select_executor_failures(executor_results, max_examples=args.max_failure_examples)
    failures_path = args.output_path.replace(".json", "_failures.json")
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=4)
    logger.info("Wrote %d executor failures to %s", len(failures), failures_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor_results_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=2)
    parser.add_argument("--max_failure_examples", type=int, default=2)
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

    main(args)
