#!/bin/bash
# Example script to generte 100 backbones given NDMi-1G

#  --silentfile \
python energy_gradient.py --steps 10 \
  --prefix-outfile ndm1_example \
  --output-path ndm1_example \
  --desired-motif-start-position 4 \
  --out-size 8 \
  --freeze-residues 1004 1005 \
  --batch-size 10000 \
  --inpaint \
  --inpaint-dist 0.1 \
  --scaffold-num 100 \
  --pdb-input ../tests/6xbf_nochanges.pdb \
  --skip-individual \
  --peptide-chain I \
  --complex-output
