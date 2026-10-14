#!/usr/bin/env python3
"""Export score data and generated peptides from a CyclicCAE silent file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_score_table(silent_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    with silent_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("SCORE:"):
                continue
            fields = line.split()[1:]
            if "description" in fields:
                header = fields
            elif header is not None and len(fields) == len(header):
                rows.append(dict(zip(header, fields)))
    if header is None or not rows:
        raise RuntimeError(f"No Rosetta score table found in {silent_path}")
    return header, rows


def select_structure_score(
    rows: list[dict[str, str]],
    *,
    tag: str | None = None,
    rank_by: str = "motif-rmsd",
) -> dict[str, str]:
    """Select a silent-file score row by tag or benchmark ranking criterion."""
    if tag is not None:
        try:
            return next(row for row in rows if row["description"] == tag)
        except StopIteration as error:
            raise ValueError(f"silent-file tag not found: {tag}") from error
    if rank_by == "motif-rmsd":
        return min(
            rows,
            key=lambda row: (
                float(row["motif_bbHeavy_RMSD"]),
                float(row["score"]),
            ),
        )
    if rank_by == "score":
        return min(rows, key=lambda row: float(row["score"]))
    raise ValueError(f"unknown structure ranking: {rank_by}")


def export_structure_from_silent(
    silent_path: Path,
    *,
    output_path: Path | None = None,
    tag: str | None = None,
    rank_by: str = "motif-rmsd",
    full_pose: bool = False,
) -> tuple[Path, dict[str, str]]:
    """Export one selected structure from a scored Rosetta silent file."""
    import pyrosetta

    silent_path = Path(silent_path)
    _, rows = parse_score_table(silent_path)
    selected = select_structure_score(rows, tag=tag, rank_by=rank_by)
    selected_tag = selected["description"]
    output_path = Path(output_path) if output_path else silent_path.with_name(
        f"{selected_tag}.pdb"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pyrosetta.init("-mute all")
    for pose in pyrosetta.io.poses_from_silent(str(silent_path)):
        if pose.pdb_info().name() != selected_tag:
            continue
        output_pose = pose if full_pose else pose.split_by_chain(1)
        output_pose.dump_pdb(str(output_path))
        return output_path, selected
    raise RuntimeError(f"Could not find {selected_tag!r} in {silent_path}")


def main() -> None:
    import pyrosetta

    parser = argparse.ArgumentParser()
    parser.add_argument("silent_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--review-count", type=int, default=15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = args.output_dir / "peptides"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    score_fields, rows = parse_score_table(args.silent_file)
    by_tag = {row["description"]: row for row in rows}

    pyrosetta.init("-mute all")
    extracted = 0
    for pose in pyrosetta.io.poses_from_silent(str(args.silent_file)):
        tag = pose.pdb_info().name()
        if tag not in by_tag:
            raise RuntimeError(f"Silent tag {tag!r} is missing from score table")
        peptide = pose.split_by_chain(1)
        pdb_name = f"{tag}.pdb"
        peptide.dump_pdb(str(pdb_dir / pdb_name))
        row = by_tag[tag]
        row["peptide_sequence"] = peptide.sequence()
        row["peptide_residue_names"] = "-".join(
            peptide.residue(i).name3() for i in range(1, peptide.total_residue() + 1)
        )
        row["peptide_pdb"] = f"peptides/{pdb_name}"
        extracted += 1

    if extracted != len(rows):
        raise RuntimeError(f"Extracted {extracted} poses for {len(rows)} score rows")

    total_ranked = sorted(rows, key=lambda row: float(row["score"]))
    motif_ranked = sorted(
        rows,
        key=lambda row: (float(row["motif_bbHeavy_RMSD"]), float(row["score"])),
    )
    total_rank = {row["description"]: rank for rank, row in enumerate(total_ranked, 1)}
    motif_rank = {row["description"]: rank for rank, row in enumerate(motif_ranked, 1)}
    review_tags = {row["description"] for row in motif_ranked[: args.review_count]}

    for row in rows:
        tag = row["description"]
        row["rank_total_score"] = str(total_rank[tag])
        row["rank_motif_bb_RMSD"] = str(motif_rank[tag])
        row["selected_for_review"] = "yes" if tag in review_tags else "no"

    extra_fields = [
        "peptide_sequence",
        "peptide_residue_names",
        "peptide_pdb",
        "rank_total_score",
        "rank_motif_bb_RMSD",
        "selected_for_review",
    ]
    csv_path = args.output_dir / "ndm1_scored_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields + extra_fields)
        writer.writeheader()
        writer.writerows(rows)

    review_path = args.output_dir / "review_tags.txt"
    review_path.write_text(
        "\n".join(row["description"] for row in motif_ranked[: args.review_count]) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Extracted {extracted} peptides to {pdb_dir}")
    print(f"Selected {len(review_tags)} review structures in {review_path}")


if __name__ == "__main__":
    main()
