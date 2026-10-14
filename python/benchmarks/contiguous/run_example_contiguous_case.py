#!/usr/bin/env python3
"""Run one enumerated contiguous-motif case from a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

from benchmarks.contiguous.summarize_example_contiguous import process_benchmark_case
from benchmarks.utilities.summary_utils import upsert_benchmark_summaries


def run_energy_gradient(
    case: dict[str, str],
    output_root: Path,
    scaffold_num: int = 100,
    batch_size: int = 10000,
    steps: int = 10,
) -> int:
    """Run energy_gradient.py for one manifest case and return its exit code."""
    run_id = case["run_id"]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_residues = case["freeze_residues"].split()
    command = [
        os.environ.get("CYCLICCAE_PYTHON", "python"),
        "python/energy_gradient.py",
        "--steps", str(steps),
        "--prefix-outfile", run_id,
        "--output-path", str(output_dir),
        "--desired-motif-start-position", case["desired_motif_start_position"],
        "--out-size", case["peptide_length"],
        "--freeze-residues", *freeze_residues,
        "--batch-size", str(batch_size),
        "--inpaint",
        "--inpaint-dist", "0.1",
        "--scaffold-num", str(scaffold_num),
        "--pdb-input", case["pdb_input"],
        "--weights-load", case["weights_load"],
        "--silentfile",
        "--skip-individual",
        "--peptide-chain", case["peptide_chain"],
        "--score-outputs",
        "--gpu",
    ]
    if case["pdb_id"] == "7BTI":
        command.append("--complex-output")
    command_path = output_dir / "command.txt"
    log_path = output_dir / "run.log"
    exit_code_path = output_dir / "exit_code.txt"
    command_path.write_text(" ".join(command) + "\n")

    print(f"[{run_id}] Starting benchmark case", flush=True)
    print(f"[{run_id}] Output directory: {output_dir}", flush=True)
    print(f"[{run_id}] Detailed log: {log_path}", flush=True)
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    exit_code_path.write_text(f"{completed.returncode}\n")

    if completed.returncode == 0:
        result = "Completed successfully"
    elif completed.returncode == 3:
        result = "Completed without finding a valid starting point"
    else:
        result = "Failed"
    print(f"[{run_id}] {result} (exit code {completed.returncode})", flush=True)
    print(f"[{run_id}] Results: {output_dir}", flush=True)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True, help="Zero-based data-row index")
    parser.add_argument("--scaffold-num", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/example_contiguous"))
    parser.add_argument("--summary-file", type=Path, default=Path("outputs/summary.csv"))
    args = parser.parse_args()

    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= args.case_index < len(rows):
        raise SystemExit(f"case index {args.case_index} outside 0..{len(rows)-1}")

    case = rows[args.case_index]
    return_code = run_energy_gradient(
        case,
        args.output_root,
        scaffold_num=args.scaffold_num,
        batch_size=args.batch_size,
        steps=args.steps,
    )
    summary, _ = process_benchmark_case(case, args.output_root)
    upsert_benchmark_summaries(args.summary_file, [summary])
    print(f"[{case['run_id']}] Updated summary: {args.summary_file}", flush=True)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
