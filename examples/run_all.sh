#!/usr/bin/env bash
# End-to-end concept-shift pipeline (single machine; step 1 needs a GPU).
# Usage: bash examples/run_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"

echo "== step 0: preprocess =="
$PY scripts/0_preprocess/prepare_data.py

echo "== step 1: AlphaGenome K562 baseline (one prediction per gene) =="
$PY scripts/1_baseline/run_ag_k562_baseline.py

echo "== step 2: assemble concept-shift table =="
$PY scripts/2_assemble/build_concept_shift_table.py

echo "== step 3: mandatory controls (within-state Spearman + Wasserstein) =="
$PY scripts/3_controls/run_controls.py

echo "== step 4: plots =="
$PY scripts/plot_concept_shift.py

echo "== done. Artifacts in ./out, plots in ./results/plots =="
