"""Expose the benchmark package when running commands from the repository root."""

from pathlib import Path


_SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "python" / "benchmarks"
if _SOURCE_PACKAGE.is_dir():
    __path__.append(str(_SOURCE_PACKAGE))
