# CyclicCAE benchmarks

This package contains experiment orchestration and result-export code. It is
kept separate from the model and inpainting implementation under `python/` so
benchmark-specific assumptions do not become part of the core runtime.

Run commands from the repository root.

## NDM1 contiguous benchmark

```bash
bash python/benchmarks/launchers/run_ndm1_contiguous.sh
python python/benchmarks/contiguous/summarize_ndm1_contiguous.py
```

## Repository example-peptide benchmark

```bash
python python/benchmarks/contiguous/prepare_example_peptide_cases.py
python python/benchmarks/contiguous/run_example_contiguous_case.py --help
python python/benchmarks/contiguous/summarize_example_contiguous.py --help
```

Cluster launchers are in `launchers/`. Their site-specific environment and
filesystem paths should be reviewed before submission.

## Utilities

`utilities/export_scored_silent.py` extracts peptide PDB files, scores, and
review selections from a scored Rosetta silent file.

Generated structures and reports belong under `outputs/`; they are not Python
package source.
