#!/usr/bin/env python3
"""Summarize manifest-driven contiguous motif inpainting runs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


SCORE_FIELDS = [
    "score", "fa_atr", "fa_rep", "fa_sol", "fa_intra_rep",
    "fa_intra_sol_xover4", "lk_ball_wtd", "fa_elec", "pro_close",
    "hbond_sr_bb", "hbond_lr_bb", "hbond_bb_sc", "hbond_sc", "dslf_fa13",
    "atom_pair_constraint", "coordinate_constraint", "angle_constraint",
    "dihedral_constraint", "metalbinding_constraint", "omega", "fa_dun",
    "p_aa_pp", "yhh_planarity", "ref", "chainbreak", "rama_prepro",
    "PeptideSize", "bb_hbonds", "defaultscorename", "design_bb_hbonds",
    "motif_allatom_nothydrogens_RMSD", "motif_bbHeavy_RMSD", "shapeComp",
    "description",
]


def parse_scores(path: Path) -> list[dict[str, str]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("SCORE:"):
                continue
            fields = line.split()[1:]
            if "description" in fields:
                header = fields
            elif header is not None and len(fields) == len(header):
                rows.append(dict(zip(header, fields)))
    return rows


def percentile(values: list[float], fraction: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timestamp(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))

    summaries: list[dict[str, object]] = []
    per_output: list[dict[str, object]] = []
    for case in cases:
        run_id = case["run_id"]
        case_dir = args.run_root / run_id
        exit_path = case_dir / "exit_code.txt"
        exit_code = exit_path.read_text().strip() if exit_path.exists() else ""
        rows = parse_scores(case_dir / f"{run_id}.silent")
        status = "PASS" if exit_code == "0" and rows else "FAIL"
        log = (case_dir / "run.log").read_text(errors="ignore") if (case_dir / "run.log").exists() else ""
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
        rank_bb = {row["description"]: rank for rank, row in enumerate(
            sorted(rows, key=lambda r: (float(r["motif_bbHeavy_RMSD"]), float(r["score"]))), 1)}
        rank_score = {row["description"]: rank for rank, row in enumerate(
            sorted(rows, key=lambda r: float(r["score"])), 1)}

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
        for row in rows:
            per_output.append({
                **common,
                "rank_within_motif_bb_rmsd": rank_bb[row["description"]],
                "rank_within_motif_score": rank_score[row["description"]],
                **{field: row.get(field, "") for field in SCORE_FIELDS},
                **identity,
            })

        summaries.append({
            **common,
            "exit_code": exit_code,
            "started_utc": timestamp(case_dir / "command.txt"),
            "finished_utc": timestamp(exit_path),
            "output_count": len(rows),
            "bb_rmsd_min": min(bb) if bb else "",
            "bb_rmsd_median": statistics.median(bb) if bb else "",
            "bb_rmsd_p90": percentile(bb, .9),
            "bb_rmsd_mean": statistics.fmean(bb) if bb else "",
            "bb_rmsd_fraction_le_0.10": sum(x <= .10 for x in bb) / len(bb) if bb else "",
            "bb_rmsd_fraction_le_0.25": sum(x <= .25 for x in bb) / len(bb) if bb else "",
            "bb_rmsd_fraction_le_0.50": sum(x <= .50 for x in bb) / len(bb) if bb else "",
            "heavy_rmsd_min": min(heavy) if heavy else "",
            "heavy_rmsd_median": statistics.median(heavy) if heavy else "",
            "heavy_rmsd_p90": percentile(heavy, .9),
            "score_min": min(scores) if scores else "",
            "score_median": statistics.median(scores) if scores else "",
            "score_mean": statistics.fmean(scores) if scores else "",
            **identity,
        })

    summary_fields = [
        "peptide_name", "motif_length", "motif_start", "freeze_residues", "motif_type",
        "status", "exit_code", "started_utc", "finished_utc", "output_count",
        "bb_rmsd_min", "bb_rmsd_median", "bb_rmsd_p90", "bb_rmsd_mean",
        "bb_rmsd_fraction_le_0.10", "bb_rmsd_fraction_le_0.25",
        "bb_rmsd_fraction_le_0.50", "heavy_rmsd_min", "heavy_rmsd_median",
        "heavy_rmsd_p90", "score_min", "score_median", "score_mean",
        "pdb_id", "example_peptide_name", "peptide_chain", "peptide_length",
        "permutation_name", "failure_reason",
    ]
    output_fields = [
        "peptide_name", "motif_length", "motif_start", "freeze_residues", "motif_type",
        "status", "rank_within_motif_bb_rmsd", "rank_within_motif_score", *SCORE_FIELDS,
        "pdb_id", "example_peptide_name", "peptide_chain", "peptide_length",
        "permutation_name", "failure_reason",
    ]
    for filename, fields, rows in (
        ("example_contiguous_summary.csv", summary_fields, summaries),
        ("example_contiguous_per_output.csv", output_fields, per_output),
    ):
        path = args.output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
