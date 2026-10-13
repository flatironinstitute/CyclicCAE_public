# CyclicCAE benchmarks

This package contains experiment orchestration and result-export code. It is
kept separate from the model and inpainting implementation under `python/` so
benchmark-specific assumptions do not become part of the core runtime.

Run commands from the repository root.

## Input structures

The example benchmark expects deposited structures under `pdbs/`. PDB files
are generated inputs and are intentionally not committed. Download the seven
standard filenames from RCSB and save the 6XBF structure under the historical
name expected by the inference example:

```bash
mkdir -p pdbs
for id in 1skl 6axi 6azg 6be9 6ucx 6wi3 7bti; do
  curl -L "https://files.rcsb.org/download/${id}.pdb" -o "pdbs/${id}.pdb"
done
curl -L https://files.rcsb.org/download/6xbf.pdb -o pdbs/6xbf_nochanges.pdb
```

The preparation command below validates these inputs and creates any
first-model or peptide-only derivatives needed by the manifests.

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

Run the cross-package review exporter as a module so its shared utility import
is resolvable:

```bash
PYTHONPATH=python python -m benchmarks.contiguous.export_contiguous_review \
  outputs/contiguous_benchmark outputs/contiguous_benchmark/review20
```

Generated structures and reports belong under `outputs/`; they are not Python
package source.
