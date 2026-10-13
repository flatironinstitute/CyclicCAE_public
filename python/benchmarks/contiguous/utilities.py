"""Shared file and score-table helpers for contiguous-motif benchmarks."""

from __future__ import annotations

import math
from collections import OrderedDict
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


def get_model_count_and_residues(
    path: Path,
) -> tuple[int, dict[str, list[tuple[str, str]]]]:
    """Return the model count and ordered polymer-like residues in model one."""
    chains: dict[str, OrderedDict[tuple[str, str], tuple[str, set[str]]]] = {}
    model_count = 0
    in_first_model = True
    saw_model = False
    with path.open(errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                model_count += 1
                saw_model = True
                in_first_model = model_count == 1
                continue
            if record == "ENDMDL" and in_first_model:
                in_first_model = False
                continue
            if record not in {"ATOM", "HETATM"} or not in_first_model:
                continue
            atom_name = line[12:16].strip()
            if atom_name not in {"N", "CA", "C", "O"}:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            chain = line[21].strip() or "_"
            key = (line[22:26].strip(), line[26].strip())
            residue_name = line[17:20].strip()
            residues = chains.setdefault(chain, OrderedDict())
            if key not in residues:
                residues[key] = (residue_name, set())
            residues[key][1].add(atom_name)
    if not saw_model:
        model_count = 1
    return model_count, {
        chain: [
            (f"{number}{insertion_code}", residue_name)
            for (number, insertion_code), (residue_name, atoms) in residues.items()
            if {"N", "CA", "C"}.issubset(atoms)
        ]
        for chain, residues in chains.items()
    }


def create_single_model_pdb(source: Path, destination: Path) -> None:
    """Create a PDB containing headers and coordinates from the first model."""
    output = []
    saw_model = False
    in_first_model = False
    with source.open(errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                if saw_model:
                    break
                saw_model = True
                in_first_model = True
                continue
            if record == "ENDMDL" and in_first_model:
                break
            if not saw_model or in_first_model:
                output.append(line)
    if not output or not saw_model:
        raise ValueError(f"{source} does not contain a MODEL record")
    output.append("END\n")
    destination.write_text("".join(output))


def create_pdb_with_selected_residues(
    source: Path,
    destination: Path,
    chain: str,
    residue_ids: set[str],
) -> None:
    """Create a PDB containing only selected coordinates and safe headers."""
    output = []
    with source.open(errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record in {"ATOM", "HETATM", "TER"}:
                line_chain = line[21].strip() or "_"
                residue_id = f"{line[22:26].strip()}{line[26].strip()}"
                if line_chain == chain and residue_id in residue_ids:
                    output.append(line)
            elif record not in {"CONECT", "MASTER", "END"}:
                output.append(line)
    output.append("END\n")
    destination.write_text("".join(output))


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
    ).isoformat(timespec="seconds")
