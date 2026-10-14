"""Command-line entry point for CyclicCAE benchmark tooling."""

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
    return parser


def main() -> None:
    """Run the selected benchmark command."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
