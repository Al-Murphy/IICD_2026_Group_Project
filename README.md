# IICD_2026_Group_Project

Concept-shift analysis of latent spaces learned from foundation models and how
they translate across state for **Perturb-seq** data — the Replogle 2022
K562-essential CRISPRi screen crossed with a **state-blind AlphaGenome
baseline**.

## The one principle

AlphaGenome predicts expression from **DNA sequence only**. A knockdown does not
change any downstream gene's sequence, so AlphaGenome's predicted effect of any
perturbation on any downstream gene is **exactly 0**. We therefore run
AlphaGenome **once per gene** for a K562 baseline (~8,563 predictions), cache it,
and reuse it across all ~1,971 perturbations. The measured non-zero change
against that zero prediction is the **concept-shift signal**.

> There are no per-(perturbation, gene) AlphaGenome calls anywhere in this repo.

## Two consumers, one shared data module

This repo produces the AlphaGenome (sequence) side. A parallel effort computes
single-cell foundation model (**scFM**, e.g. scVI) embedding displacements on
the same screen. Both sides **must** operate on identical filtered cells,
pseudobulk, and per-perturbation relevant-gene sets — so all of that lives in
one importable, dependency-light module:

```python
from concept_shift import data
csd = data.prepare(filtered_h5ad="out/replogle_k562_filtered.h5ad")
csd.adata            # 30-cell-filtered, normalised AnnData (all 8,563 genes)
csd.pb, csd.delta    # pseudobulk mean + measured delta vs control
csd.relevant_genes   # {perturbation: [trans gene var_names]} (target + cis excluded)
csd.coords           # strand-aware hg38 TSS table
```

`concept_shift.data` imports **without** torch / AlphaGenome, so the scFM team
installs only the core deps.

## Layout

```
concept_shift/            # importable package
├── data.py               #  ⭐ shared load / filter / pseudobulk / relevant-gene sets (steps 1-7)
├── seq.py                #  hg38 genome fetch + one-hot (AlphaGenome extra)
└── ag_backbone.py        #  AlphaGenome pytorch-port K562 baseline (AlphaGenome extra)
scripts/
├── 0_preprocess/prepare_data.py            # steps 1-7 -> out/ artifacts
├── 1_baseline/run_ag_k562_baseline.py      # step 7: one prediction per gene (resumable, shardable)
├── 2_assemble/build_concept_shift_table.py # step 8: concept_shift_table (predicted delta == 0)
├── 3_controls/run_controls.py              # step 9: within-state Spearman + Wasserstein strength baseline
└── plot_concept_shift.py                   # summary plots
metadata/track_metadata.parquet             # K562 AlphaGenome track indices
tests/                                       # pytest (non-GPU, no weights, no network)
examples/                                    # run_all.sh + SLURM array template
```

## Environment (uv)

```bash
# Core (data module only — enough for the scFM team):
uv sync

# Add the AlphaGenome baseline deps (torch, pysam, tangermeme, alphagenome-pytorch):
uv sync --extra alphagenome --extra plotting
```

Python is pinned to ≥3.10 (the pipeline runs on 3.11). AlphaGenome uses the
**pytorch port with local weights** (`gtca/alphagenome_pytorch`, all-folds), not
the API — no rate limits, no key.

## Running the pipeline

```bash
# 0. Preprocess (downloads the screen; DE step is the slow part).
python scripts/0_preprocess/prepare_data.py

# 1. AlphaGenome K562 baseline — one prediction per gene (needs a GPU; resumable).
python scripts/1_baseline/run_ag_k562_baseline.py

# 2. Assemble the concept-shift table (predicted delta == 0).
python scripts/2_assemble/build_concept_shift_table.py

# 3. Mandatory controls: within-state Spearman + Wasserstein strength baseline.
python scripts/3_controls/run_controls.py

# 4. Plots.
python scripts/plot_concept_shift.py

# Or the whole thing:
bash examples/run_all.sh
```

The 1 Mb AlphaGenome baseline is best sharded on a cluster — see
[`examples/slurm_ag_baseline_array.sh`](examples/slurm_ag_baseline_array.sh).

## Mandatory controls (why this is *concept shift*, not model weakness)

- **Within-state Spearman** (step 9a): AlphaGenome baseline vs measured control
  pseudobulk across genes must be reasonably positive. If AlphaGenome can't even
  predict the in-state K562 profile, a non-zero "error" is generic weakness, not
  a shift. Restrict to genes AlphaGenome predicts acceptably in-state otherwise.
- **Wasserstein strength baseline** (step 9b): per-perturbation OT distance
  (perturbed vs control). We use **Wasserstein** (not E-distance) to match the
  OT metric the scFM side applies on the scVI latent, so the strength axis is
  directly comparable. If a scFM displacement doesn't out-predict raw
  Wasserstein distance, the foundation model adds nothing (cf. benchmark
  Fig. 5e, strength↔error r≈0.8). Point `--obsm_key X_scVI` at the scVI latent
  to compute it in exactly the scFM space.

## Outputs (`out/`)

| File | Produced by | Contents |
|---|---|---|
| `replogle_k562_filtered.h5ad` | step 0 | 30-cell-filtered, normalised AnnData (shared with scFM) |
| `pseudobulk.parquet`, `delta.parquet` | step 0 | per-perturbation mean + measured delta |
| `knockdown_check.csv` | step 0 | soft percent-of-control QC flag (never drops) |
| `coords.parquet` | step 0 | strand-aware hg38 TSS table |
| `relevant_genes.{parquet,json}` | step 0 | per-perturbation trans gene sets |
| `ag_k562_baseline.parquet` | step 1 | CAGE (primary) + RNA-seq (secondary) per gene |
| `concept_shift_table.parquet` | step 2 | `(pert, gene, measured_delta, pred_delta=0, error)` |
| `perturbation_summary.parquet` | steps 2–3 | per-pert rollup + Wasserstein distance |
| `within_state_spearman.txt`, `wasserstein_distance.parquet` | step 3 | controls |

## Tests

```bash
uv run pytest
```

Non-GPU, no model weights, no network. Covers normalisation detection, the
30-cell filter, pseudobulk/delta, percent-based knockdown QC, strand-aware
coordinates, target+cis exclusion in the relevant-gene sets, and AlphaGenome
track selection / gene-body geometry.

## License

MIT.
