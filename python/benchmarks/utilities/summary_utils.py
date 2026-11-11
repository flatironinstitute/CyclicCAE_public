"""Concurrency-safe maintenance of the unified benchmark summary."""

from __future__ import annotations

import csv
import fcntl
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


BENCHMARK_SUMMARY_FIELDS = [
    "peptide_name",
    "motif_length",
    "motif_start",
    "freeze_residues",
    "motif_type",
    "status",
    "exit_code",
    "started_utc",
    "finished_utc",
    "output_count",
    "bb_rmsd_min",
    "bb_rmsd_median",
    "bb_rmsd_p90",
    "bb_rmsd_mean",
    "bb_rmsd_fraction_le_0.10",
    "bb_rmsd_fraction_le_0.25",
    "bb_rmsd_fraction_le_0.50",
    "heavy_rmsd_min",
    "heavy_rmsd_median",
    "heavy_rmsd_p90",
    "score_min",
    "score_median",
    "score_mean",
    "pdb_id",
    "example_peptide_name",
    "peptide_chain",
    "peptide_length",
    "permutation_name",
    "failure_reason",
]


def _summary_key(row: Mapping[str, object]) -> str:
    """Return the stable case identifier used when replacing summary rows."""
    key = str(row.get("permutation_name") or row.get("peptide_name") or "")
    if not key:
        raise ValueError("benchmark summary row has no permutation_name or peptide_name")
    return key


def upsert_benchmark_summaries(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Atomically insert or replace summary rows while holding a file lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        merged: dict[str, dict[str, object]] = {}
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for existing in csv.DictReader(handle):
                    merged[_summary_key(existing)] = existing
        for row in rows:
            merged[_summary_key(row)] = dict(row)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                writer = csv.DictWriter(
                    temporary,
                    fieldnames=BENCHMARK_SUMMARY_FIELDS,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(merged.values())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
