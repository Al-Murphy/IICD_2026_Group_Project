#!/bin/bash
# SLURM array: AlphaGenome K562 baseline for the IICD concept-shift project.
#
# AlphaGenome is state-blind, so we predict ONCE per gene (8,545 genes with valid
# hg38 coords) and reuse that baseline across all 1,971 perturbations -- the
# predicted delta for every (perturbation, gene) pair is exactly 0. There are NO
# per-perturbation forward passes here.
#
# Each array task predicts genes[shard_idx::NUM_SHARDS] and writes one shard
# parquet (resumable: a re-queued task skips genes already present in its shard).
# Merge after all tasks finish (command at the bottom).
#
# Submit: sbatch /grid/koo/home/amurphy/projects/job_scripts/iicd_ag_k562_baseline_array.sh
#
#SBATCH --job-name=iicd_ag_k562_bioai
#SBATCH --output=/grid/koo/home/amurphy/projects/job_scripts/out/iicd_ag_k562_baseline_bioai_%A_%a.out
#SBATCH --error=/grid/koo/home/amurphy/projects/job_scripts/out/iicd_ag_k562_baseline_bioai_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amurphy@cshl.edu
#SBATCH --export=ALL
#SBATCH --array=0-4          # NUM_SHARDS - 1  (5 shards x ~1,709 genes each; fits QOS concurrent-job limit)

set -euo pipefail

NUM_SHARDS=5

# --export=ALL propagates the SUBMITTING shell's SLURM_* vars. If you sbatch from
# inside an interactive allocation, SLURM_TRES_PER_TASK leaks in and conflicts with
# this job's SLURM_CPUS_PER_TASK ("cpus_per_task set by two different environment
# variables"). Scrub them; this job is a single task so it needs no srun anyway.
unset SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK SLURM_EXPORT_ENV 2>/dev/null || true

echo "Job ID: ${SLURM_JOB_ID:-}  Array task: ${SLURM_ARRAY_TASK_ID:-}"
echo "Node: ${SLURM_JOB_NODELIST:-}"
echo "Start time: $(date)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-}"

# AlphaGenome pytorch port (local weights, gtca/alphagenome_pytorch all-folds).
# We do NOT use the official API -- no rate limits, no key.
cd /grid/koo/home/amurphy/projects/alphagenome-pytorch
source .venv/bin/activate

PROJ_ROOT="${PROJ_ROOT:-/grid/koo/home/amurphy/projects/IICD_2026_Group_Project}"
cd "${PROJ_ROOT}"

# concept_shift lives in this repo but is not installed into the AG venv;
# expose it on the path instead of polluting that env with pertpy/scanpy.
export PYTHONPATH="${PROJ_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:+$PYTORCH_CUDA_ALLOC_CONF,}expandable_segments:True"

mkdir -p out /grid/koo/home/amurphy/projects/job_scripts/out

COORDS="${COORDS:-${PROJ_ROOT}/out/coords.parquet}"
METADATA_PATH="${METADATA_PATH:-${PROJ_ROOT}/metadata/track_metadata.parquet}"
GENOME_CACHE="${GENOME_CACHE:-${PROJ_ROOT}/.cache}"   # hg38.fa symlinked from clg/.cache
SHARD_OUT="${PROJ_ROOT}/out/ag_k562_baseline_shard${SLURM_ARRAY_TASK_ID}of${NUM_SHARDS}.parquet"

# Preflight: fail fast with a clear message rather than 8 GPUs dying on a missing file.
[[ -f "${COORDS}" ]]        || { echo "ERROR: missing ${COORDS} (run scripts/0_preprocess/prepare_data.py)"; exit 1; }
[[ -f "${METADATA_PATH}" ]] || { echo "ERROR: missing ${METADATA_PATH}"; exit 1; }
[[ -e "${GENOME_CACHE}/hg38.fa" ]] || { echo "ERROR: missing ${GENOME_CACHE}/hg38.fa (symlink it; do not let 8 shards race to download)"; exit 1; }

python scripts/1_baseline/run_ag_k562_baseline.py \
  --coords "${COORDS}" \
  --metadata_path "${METADATA_PATH}" \
  --genome_cache "${GENOME_CACHE}" \
  --genome_build hg38 \
  --num_shards "${NUM_SHARDS}" \
  --shard_idx "${SLURM_ARRAY_TASK_ID}" \
  --output "${SHARD_OUT}" \
  --save_every 200

deactivate
echo "End time: $(date)"

# =============================================================================
# After ALL array tasks finish, merge the shards (CPU is fine). Uses the project
# venv (pandas) and reports coverage against the pseudobulk gene set:
#
#   cd /grid/koo/home/amurphy/projects/IICD_2026_Group_Project
#   .venv/bin/python - <<'PY'
#   import glob, pandas as pd
#   parts = [pd.read_parquet(f) for f in sorted(glob.glob("out/ag_k562_baseline_shard*of5.parquet"))]
#   base = pd.concat(parts).rename_axis("gene")
#   base.to_parquet("out/ag_k562_baseline.parquet")
#   pb = pd.read_parquet("out/pseudobulk.parquet")
#   covered = len(set(pb.columns) & set(base.index))
#   print(f"merged {len(base)} gene predictions from {len(parts)} shards")
#   print(f"pseudobulk coverage: {covered}/{len(pb.columns)} genes")
#   print(base.head())
#   PY
#
# Then run Wasserstein + the per-state concept-shift analysis:
#   .venv/bin/python scripts/2_wasserstein/run_wasserstein.py
#   .venv/bin/python scripts/3_analysis/run_state_shift.py
# =============================================================================
