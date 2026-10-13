#!/usr/bin/env python3
"""Export a representative set of peptides from contiguous benchmark silent files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pyrosetta

from benchmarks.utilities.export_scored_silent import parse_score_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    summary = list(
        csv.DictReader((args.benchmark_root / "contiguous_per_motif_summary.csv").open())
    )
    passing = [row["peptide_name"] for row in summary if row["status"] == "PASS"]

    ranked: dict[str, list[dict[str, str]]] = {}
    for name in passing:
        silent = args.benchmark_root / name / f"{name}.silent"
        _, rows = parse_score_table(silent)
        ranked[name] = sorted(
            rows,
            key=lambda row: (
                float(row["motif_bbHeavy_RMSD"]),
                float(row["score"]),
            ),
        )

    selected: list[tuple[str, dict[str, str], int]] = []
    rank = 0
    while len(selected) < args.count:
        added = False
        for name in passing:
            if rank < len(ranked[name]) and len(selected) < args.count:
                selected.append((name, ranked[name][rank], rank + 1))
                added = True
        if not added:
            break
        rank += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    peptide_dir = args.output_dir / "peptides"
    peptide_dir.mkdir(exist_ok=True)
    wanted = defaultdict(dict)
    for index, (name, row, within_rank) in enumerate(selected, 1):
        wanted[name][row["description"]] = (index, row, within_rank)

    pyrosetta.init("-mute all")
    manifest = []
    for name in passing:
        if name not in wanted:
            continue
        silent = args.benchmark_root / name / f"{name}.silent"
        for pose in pyrosetta.io.poses_from_silent(str(silent)):
            tag = pose.pdb_info().name()
            if tag not in wanted[name]:
                continue
            index, row, within_rank = wanted[name][tag]
            peptide = pose.split_by_chain(1)
            filename = f"review_{index:02d}_{tag}.pdb"
            peptide.dump_pdb(str(peptide_dir / filename))
            manifest.append(
                {
                    "review_index": index,
                    "peptide_name": name,
                    "tag": tag,
                    "rank_within_motif_bb_rmsd": within_rank,
                    "motif_bbHeavy_RMSD": row["motif_bbHeavy_RMSD"],
                    "motif_allatom_nothydrogens_RMSD": row[
                        "motif_allatom_nothydrogens_RMSD"
                    ],
                    "score": row["score"],
                    "pdb": f"peptides/{filename}",
                }
            )

    manifest.sort(key=lambda row: row["review_index"])
    with (args.output_dir / "review20_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Exported {len(manifest)} peptides")


if __name__ == "__main__":
    main()
