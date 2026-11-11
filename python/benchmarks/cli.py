"""Command-line interface for CyclicCAE benchmark tooling."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.utilities.structure_utils import download_benchmark_structures


def pull_structures(args: argparse.Namespace) -> None:
    """Download the deposited PDB structures used by the benchmarks."""
    for path, downloaded in download_benchmark_structures(
        args.output_dir,
        force=args.force,
    ):
        action = "Downloaded" if downloaded else "Already present"
        print(f"{action}: {path}")


def export_structure(args: argparse.Namespace) -> None:
    """Export one selected PDB from a scored Rosetta silent file."""
    from benchmarks.utilities.export_scored_silent import export_structure_from_silent

    output_path, score = export_structure_from_silent(
        args.silent_file,
        output_path=args.output,
        tag=args.tag,
        rank_by=args.rank_by,
        full_pose=args.full_pose,
    )
    print(f"Wrote: {output_path}")
    print(f"Tag: {score['description']}")
    if "motif_bbHeavy_RMSD" in score:
        print(f"Motif backbone RMSD: {score['motif_bbHeavy_RMSD']}")
    if "score" in score:
        print(f"Score: {score['score']}")


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Prepare and run CyclicCAE benchmarks.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    pull_parser = subcommands.add_parser(
        "pull-structures",
        help="download the deposited PDB structures used by the benchmarks",
    )
    pull_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pdbs"),
        help="destination directory (default: pdbs)",
    )
    pull_parser.add_argument(
        "--force",
        action="store_true",
        help="replace structures that are already present",
    )
    pull_parser.set_defaults(handler=pull_structures)

    export_parser = subcommands.add_parser(
        "export-structure",
        help="export one PDB from a scored benchmark silent file",
    )
    export_parser.add_argument("silent_file", type=Path)
    export_parser.add_argument(
        "--output",
        type=Path,
        help="output PDB path (default: selected tag beside the silent file)",
    )
    export_parser.add_argument(
        "--tag",
        help="export this exact silent-file tag instead of selecting by score",
    )
    export_parser.add_argument(
        "--rank-by",
        choices=("motif-rmsd", "score"),
        default="motif-rmsd",
        help="criterion used when --tag is omitted (default: motif-rmsd)",
    )
    export_parser.add_argument(
        "--full-pose",
        action="store_true",
        help="export the complete pose instead of peptide chain 1",
    )
    export_parser.set_defaults(handler=export_structure)
    return parser


def main() -> None:
    """Run the selected benchmark command."""
    args = build_parser().parse_args()
    args.handler(args)
