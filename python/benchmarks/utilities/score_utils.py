"""Rosetta score parsing and summary helpers for benchmarks."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path


ROSETTA_SCORE_FIELDS = [
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


def parse_rosetta_scores(path: Path) -> list[dict[str, str]]:
    """Parse score rows from a Rosetta silent file, or return an empty list."""
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


def calculate_percentile(values: list[float], fraction: float) -> float | str:
    """Return a linearly interpolated percentile, or an empty value for no data."""
    if not values:
        return ""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def get_file_modified_time_utc(path: Path) -> str:
    """Return a file's modification time as an ISO-8601 UTC timestamp."""
    if not path.exists():
        return ""
    return datetime.fromtimestamp(
        path.stat().st_mtime,
        timezone.utc,
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
