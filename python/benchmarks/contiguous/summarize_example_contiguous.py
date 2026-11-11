#!/usr/bin/env python3
"""Summarize manifest-driven contiguous motif inpainting runs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from benchmarks.utilities.score_utils import (
    ROSETTA_SCORE_FIELDS,
    calculate_percentile,
    get_file_modified_time_utc,
    parse_rosetta_scores,
)
from benchmarks.utilities.summary_utils import (
    BENCHMARK_SUMMARY_FIELDS,
    upsert_benchmark_summaries,
)


def process_benchmark_case(
    case: dict[str, str],
    run_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build one case summary and its per-output score records."""
    run_id = case["run_id"]
    case_dir = run_root / run_id
    exit_path = case_dir / "exit_code.txt"
    exit_code = exit_path.read_text().strip() if exit_path.exists() else ""
    rows = parse_rosetta_scores(case_dir / f"{run_id}.silent")
    status = "PASS" if exit_code == "0" and rows else "FAIL"
    log_path = case_dir / "run.log"
    log = log_path.read_text(errors="ignore") if log_path.exists() else ""
    if status == "PASS":
        failure_reason = ""
    elif exit_code == "3" or "No valid starting points" in log:
        failure_reason = "NO_MATCH"
    elif not exit_code:
        failure_reason = "NOT_RUN"
    else:
        failure_reason = "ERROR"

    bb = [float(row["motif_bbHeavy_RMSD"]) for row in rows]
    heavy = [float(row["motif_allatom_nothydrogens_RMSD"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    rank_bb = {
        row["description"]: rank
        for rank, row in enumerate(
            sorted(
                rows,
                key=lambda score: (
                    float(score["motif_bbHeavy_RMSD"]),
                    float(score["score"]),
                ),
            ),
            1,
        )
    }
    rank_score = {
        row["description"]: rank
        for rank, row in enumerate(
            sorted(rows, key=lambda score: float(score["score"])),
            1,
        )
    }

    identity = {
        "pdb_id": case["pdb_id"],
        "example_peptide_name": case["peptide_name"],
        "peptide_chain": case["peptide_chain"],
        "peptide_length": case["peptide_length"],
        "permutation_name": run_id,
        "failure_reason": failure_reason,
    }
    common = {
        "peptide_name": run_id,
        "motif_length": case["motif_length"],
        "motif_start": case["motif_start_index"],
        "freeze_residues": case["freeze_residues"],
        "motif_type": case["contiguity"],
        "status": status,
    }
    per_output = [
        {
            **common,
            "rank_within_motif_bb_rmsd": rank_bb[row["description"]],
            "rank_within_motif_score": rank_score[row["description"]],
            **{field: row.get(field, "") for field in ROSETTA_SCORE_FIELDS},
            **identity,
        }
        for row in rows
    ]

    summary = {
        **common,
        "exit_code": exit_code,
        "started_utc": get_file_modified_time_utc(case_dir / "command.txt"),
        "finished_utc": get_file_modified_time_utc(exit_path),
        "output_count": len(rows),
        "bb_rmsd_min": min(bb) if bb else "",
        "bb_rmsd_median": statistics.median(bb) if bb else "",
        "bb_rmsd_p90": calculate_percentile(bb, .9),
        "bb_rmsd_mean": statistics.fmean(bb) if bb else "",
        "bb_rmsd_fraction_le_0.10": sum(x <= .10 for x in bb) / len(bb) if bb else "",
        "bb_rmsd_fraction_le_0.25": sum(x <= .25 for x in bb) / len(bb) if bb else "",
        "bb_rmsd_fraction_le_0.50": sum(x <= .50 for x in bb) / len(bb) if bb else "",
        "heavy_rmsd_min": min(heavy) if heavy else "",
        "heavy_rmsd_median": statistics.median(heavy) if heavy else "",
        "heavy_rmsd_p90": calculate_percentile(heavy, .9),
        "score_min": min(scores) if scores else "",
        "score_median": statistics.median(scores) if scores else "",
        "score_mean": statistics.fmean(scores) if scores else "",
        **identity,
    }
    return summary, per_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, default=Path("outputs/summary.csv"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))

    summaries: list[dict[str, object]] = []
    per_output: list[dict[str, object]] = []
    for case in cases:
        summary, case_outputs = process_benchmark_case(case, args.run_root)
        summaries.append(summary)
        per_output.extend(case_outputs)

    output_fields = [
        "peptide_name", "motif_length", "motif_start", "freeze_residues", "motif_type",
        "status", "rank_within_motif_bb_rmsd", "rank_within_motif_score",
        *ROSETTA_SCORE_FIELDS,
        "pdb_id", "example_peptide_name", "peptide_chain", "peptide_length",
        "permutation_name", "failure_reason",
    ]
    for filename, fields, rows in (
        ("example_contiguous_summary.csv", BENCHMARK_SUMMARY_FIELDS, summaries),
        ("example_contiguous_per_output.csv", output_fields, per_output),
    ):
        path = args.output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}")

    upsert_benchmark_summaries(args.summary_file, summaries)
    print(f"Updated {len(summaries)} rows in {args.summary_file}")


if __name__ == "__main__":
    main()
