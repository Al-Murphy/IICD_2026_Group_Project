#!/usr/bin/env bash
# SLURM array template for the AlphaGenome K562 baseline (step 1).
# Each array task predicts genes[shard_idx::num_shards] and writes a shard parquet;
# the resume logic makes re-queued tasks cheap. Set partition/QOS/GPU for your cluster.
#
#   sbatch --array=0-7 examples/slurm_ag_baseline_array.sh
#
# After all shards finish, concatenate the shard parquets into one baseline:
#   .venv/bin/python - <<'PY'
#   import glob, pandas as pd
#   parts = [pd.read_parquet(f) for f in glob.glob("out/ag_k562_baseline_shard*of8.parquet")]
#   pd.concat(parts).to_parquet("out/ag_k562_baseline.parquet")
#   PY

#SBATCH --job-name=ag_k562_baseline
#SBATCH --partition=gpu             # <- set for your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ag_baseline_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

NUM_SHARDS=8
.venv/bin/python scripts/1_baseline/run_ag_k562_baseline.py \
    --num_shards "$NUM_SHARDS" \
    --shard_idx "$SLURM_ARRAY_TASK_ID" \
    --output "out/ag_k562_baseline_shard${SLURM_ARRAY_TASK_ID}of${NUM_SHARDS}.parquet"
