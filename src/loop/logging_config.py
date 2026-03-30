"""Shared logging configuration for per-problem debug log files."""

import logging
from pathlib import Path


def setup_problem_logger(log_file: Path) -> None:
    """Attach a FileHandler to the root logger for a specific problem's debug log.

    Removes any existing FileHandlers first to prevent handler accumulation
    when called multiple times in the same process (e.g. sequential problem
    runs in run_experiment.py).
    """
    root = logging.getLogger()

    for h in root.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            root.removeHandler(h)

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
