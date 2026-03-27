"""Convert candidates.csv (with profile data) to profile_results.csv.

The executor already profiled each kernel. select_candidates.py passes the
profile metadata through. This script just reformats the CSV into the shape
the next iteration's planner expects. No re-profiling needed.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    # Kept for CLI compatibility but unused — profiling already happened in executor
    parser.add_argument("--profile_mode", type=str, default="cycle_accurate")
    parser.add_argument("--rel_tol", type=float, default=2e-5)
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

    df = pd.read_csv(args.candidates_path)
    assert "profile" in df.columns, (
        f"candidates.csv missing 'profile' column. Columns: {list(df.columns)}"
    )
    df.to_csv(args.output_path, index=False)
