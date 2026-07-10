#!/usr/bin/env bash
# End-to-end concept-shift pipeline. Step 1 needs a GPU (see the SLURM templates).
# Usage: bash examples/run_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"

echo "== step 0: preprocess (downloads the 1.44 GB scPerturb h5ad on first run) =="
$PY scripts/0_preprocess/prepare_data.py

echo "== step 1: AlphaGenome K562 baseline -- one prediction per gene (GPU) =="
$PY scripts/1_baseline/run_ag_k562_baseline.py

echo "== step 2: per-perturbation Wasserstein (OT) distance =="
$PY scripts/2_wasserstein/run_wasserstein.py

echo "== step 3: per-state Spearman + matched-null concept shift =="
$PY scripts/3_analysis/run_state_shift.py

echo "== done. Tables in ./out, plots in ./results/plots =="
