#!/usr/bin/env python3
"""Aggregate scored CyclicCAE contiguous-motif benchmark silent files."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def planned_nonwrap_cases() -> list[dict[str, object]]:
    cases = []
    for length in (2, 3, 4):
        for start in range(1, 8 - length + 2):
            residues = list(range(1000 + start, 1000 + start + length))
            cases.append(
                {
                    "peptide_name": f"NDM1_L{length}_S{1000 + start}",
                    "motif_length": length,
                    "motif_start": 1000 + start,
                    "freeze_residues": " ".join(map(str, residues)),
                    "motif_type": "Contiguous",
                }
            )
    return cases


def parse_scores(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    header = None
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("SCORE:"):
                continue
            fields = line.split()[1:]
            if "description" in fields:
                header = fields
            elif header is not None and len(fields) == len(header):
                rows.append(dict(zip(header, fields)))
    return header or [], rows


def read_config(path: Path) -> dict[str, str]:
    result = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
    return result


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark_root", type=Path, nargs="?", default=Path("outputs/contiguous_benchmark")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.benchmark_root
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    summaries: list[dict[str, object]] = []
    score_fields: list[str] = []

    for case in planned_nonwrap_cases():
        name = str(case["peptide_name"])
        case_dir = args.benchmark_root / name
        config = read_config(case_dir / "run_config.txt")
        silent = case_dir / f"{name}.silent"
        fields, rows = parse_scores(silent) if silent.exists() else ([], [])
        if fields and not score_fields:
            score_fields = fields

        status = "PASS" if config.get("exit_code") == "0" and rows else "FAIL"
        bb = [float(row["motif_bbHeavy_RMSD"]) for row in rows]
        heavy = [float(row["motif_allatom_nothydrogens_RMSD"]) for row in rows]
        total = [float(row["score"]) for row in rows]

        ranked_bb = {
            row["description"]: rank
            for rank, row in enumerate(
                sorted(rows, key=lambda r: (float(r["motif_bbHeavy_RMSD"]), float(r["score"]))), 1
            )
        }
        ranked_score = {
            row["description"]: rank
            for rank, row in enumerate(sorted(rows, key=lambda r: float(r["score"])), 1)
        }
        for row in rows:
            all_rows.append(
                {
                    **{key: str(value) for key, value in case.items()},
                    "status": status,
                    "rank_within_motif_bb_rmsd": str(ranked_bb[row["description"]]),
                    "rank_within_motif_score": str(ranked_score[row["description"]]),
                    **row,
                }
            )

        summary = {
            **case,
            "status": status,
            "exit_code": config.get("exit_code", ""),
            "started_utc": config.get("started_utc", ""),
            "finished_utc": config.get("finished_utc", ""),
            "output_count": len(rows),
            "bb_rmsd_min": min(bb) if bb else "",
            "bb_rmsd_median": statistics.median(bb) if bb else "",
            "bb_rmsd_p90": percentile(bb, 0.90) if bb else "",
            "bb_rmsd_mean": statistics.fmean(bb) if bb else "",
            "bb_rmsd_fraction_le_0.10": sum(x <= 0.10 for x in bb) / len(bb) if bb else "",
            "bb_rmsd_fraction_le_0.25": sum(x <= 0.25 for x in bb) / len(bb) if bb else "",
            "bb_rmsd_fraction_le_0.50": sum(x <= 0.50 for x in bb) / len(bb) if bb else "",
            "heavy_rmsd_min": min(heavy) if heavy else "",
            "heavy_rmsd_median": statistics.median(heavy) if heavy else "",
            "heavy_rmsd_p90": percentile(heavy, 0.90) if heavy else "",
            "score_min": min(total) if total else "",
            "score_median": statistics.median(total) if total else "",
            "score_mean": statistics.fmean(total) if total else "",
        }
        summaries.append(summary)

    metadata_fields = [
        "peptide_name", "motif_length", "motif_start", "freeze_residues", "motif_type", "status"
    ]
    rank_fields = ["rank_within_motif_bb_rmsd", "rank_within_motif_score"]
    per_output_path = output_dir / "contiguous_per_output.csv"
    with per_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields + rank_fields + score_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output_dir / "contiguous_per_motif_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(all_rows)} output rows to {per_output_path}")
    print(f"Wrote {len(summaries)} motif rows to {summary_path}")


if __name__ == "__main__":
    main()
