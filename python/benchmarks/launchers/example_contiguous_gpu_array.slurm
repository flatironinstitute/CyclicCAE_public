#!/usr/bin/env bash
#SBATCH --job-name=cycliccae-contig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/example_contiguous/slurm/%A_%a.out
#SBATCH --error=outputs/example_contiguous/slurm/%A_%a.err

set -euo pipefail

repo_dir="/home/rmcl/parisahlab/rmcl/cyclic-cae/CyclicCAE"
env_dir="/home/rmcl/miniconda3/envs/cycliccae-dev"
cd "$repo_dir"
mkdir -p outputs/example_contiguous/slurm

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CYCLICCAE_PYTHON="$env_dir/bin/python"

"$env_dir/bin/python" -m benchmarks.contiguous.run_example_contiguous_case \
  --manifest "${CYCLICCAE_MANIFEST:-outputs/example_peptides/example_peptide_contiguous_run_cases.csv}" \
  --case-index "$SLURM_ARRAY_TASK_ID" \
  --output-root "${CYCLICCAE_OUTPUT_ROOT:-outputs/example_contiguous}" \
  --scaffold-num "${CYCLICCAE_SCAFFOLD_NUM:-100}" \
  --batch-size "${CYCLICCAE_BATCH_SIZE:-10000}" \
  --steps "${CYCLICCAE_STEPS:-10}"
