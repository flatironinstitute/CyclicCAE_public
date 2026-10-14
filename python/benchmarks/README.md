# CyclicCAE Benchmark

Exercises CyclicCAE using known structures to generate peptides from known
starting structures.

Run all commands from the repository root. The examples below assume the
project environment is available at `.env/` and has been activated:

```bash
source .env/bin/activate
```

## How to run

Download the deposited structures required by the benchmark:

```bash
python -m benchmarks pull-structures
```

Existing valid files are left unchanged. Pass `--force` to download them
again. The command saves the structures under `pdbs/`, including 6XBF as
`pdbs/6xbf_nochanges.pdb` for compatibility with the original example.

Prepare the structure inventory and contiguous-motif case manifests:

```bash
python -m benchmarks.contiguous.prepare_example_peptide_cases
```

Run one case by its zero-based row number. Use the six-row pilot manifest
first; the production manifest contains 90 cases.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m benchmarks.contiguous.run_example_contiguous_case \
  --manifest outputs/example_peptides/example_peptide_contiguous_pilot_cases.csv \
  --case-index 0 \
  --output-root outputs/example_contiguous_pilot
```

On a Slurm cluster, review the repository and environment paths in
`python/benchmarks/launchers/example_contiguous_gpu_array.slurm`, then launch
the pilot or production array:

```bash
sbatch --array=0-5 \
  --export=ALL,CYCLICCAE_MANIFEST=outputs/example_peptides/example_peptide_contiguous_pilot_cases.csv,CYCLICCAE_OUTPUT_ROOT=outputs/example_contiguous_pilot \
  python/benchmarks/launchers/example_contiguous_gpu_array.slurm

sbatch --array=0-89 \
  python/benchmarks/launchers/example_contiguous_gpu_array.slurm
```

Summarize completed production runs:

```bash
python -m benchmarks.contiguous.summarize_example_contiguous \
  --manifest outputs/example_peptides/example_peptide_contiguous_run_cases.csv \
  --run-root outputs/example_contiguous \
  --output-dir outputs/example_contiguous/reports
```

Export the generated peptide with the lowest motif backbone RMSD from one
case's scored silent file:

```bash
python -m benchmarks export-structure \
  outputs/example_contiguous_pilot/6AZG_L2_S01/6AZG_L2_S01.silent
```

By default, this selects the output with the lowest motif backbone RMSD,
extracts peptide chain 1, and writes `<structure_tag>.pdb` beside the silent
file. The command prints the selected tag, motif RMSD, score, and output path.

Other selection and export options are:

```bash
# Export the lowest-energy output instead.
python -m benchmarks export-structure CASE.silent --rank-by score

# Export an exact Rosetta silent-file tag.
python -m benchmarks export-structure CASE.silent --tag TAG

# Retain the complete pose, including every chain in a complex.
python -m benchmarks export-structure CASE.silent --full-pose

# Choose the output PDB path.
python -m benchmarks export-structure CASE.silent --output best.pdb
```

The selection rules belong to the benchmark CLI, while the reusable
silent-file extraction code lives in `benchmarks.utilities.export_scored_silent`.

The original NDM1 contiguous benchmark can be run separately:

```bash
bash python/benchmarks/launchers/run_ndm1_contiguous.sh pilot
python -m benchmarks.contiguous.summarize_ndm1_contiguous
```

## Description of output

Case manifests and input checks are written to `outputs/example_peptides/`.
Each run writes its command, log, exit code, and scored silent file beneath
`outputs/example_contiguous/<run_id>/`. Each completed case is inserted or
updated in `outputs/summary.csv`. The summarizer also produces one CSV of
case-level statistics and one CSV containing every scored output.
