#!/usr/bin/env python3
"""
Plot concept-shift summaries (clg plotting style).

Panels
------
1. Distribution of measured |delta| across all (pert, gene) trans pairs -- the
   concept-shift magnitude AlphaGenome misses entirely (predicted delta == 0).
2. Per-perturbation mean error vs Wasserstein (OT) distance -- the benchmark
   strength axis a scFM displacement must beat (expect a strong positive
   relationship). Wasserstein matches the OT metric applied on the scVI latent.
3. Within-state check: AlphaGenome K562 baseline vs measured control pseudobulk.

Usage
-----
    python scripts/plot_concept_shift.py
    python scripts/plot_concept_shift.py --output results/plots/concept_shift.png
"""

import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot concept-shift summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--concept_table", default="./out/concept_shift_table.parquet")
    p.add_argument("--sample_rows", type=int, default=2_000_000,
                   help="Subsample rows for the histogram (16.8M rows plot slowly).")
    p.add_argument("--summary", default="./out/perturbation_summary.parquet")
    p.add_argument("--baseline", default="./out/ag_k562_baseline.parquet")
    p.add_argument("--pseudobulk", default="./out/pseudobulk.parquet")
    p.add_argument("--baseline_proxy", default="rna_pred", choices=["cage_pred", "rna_pred"],
                   help="rna_pred (gene body) is primary: it beats cage_pred on the "
                        "within-state control (rho 0.67 vs 0.34), since the measured "
                        "observable is RNA-seq expression.")
    p.add_argument("--ctrl", default="control")
    p.add_argument("--output", default="./results/plots/concept_shift.png")
    return p.parse_args()


def _style():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    warm, cool = "#d1495b", "#2e6f95"
    return plt, sns, warm, cool


def main():
    args = parse_args()
    plt, sns, warm, cool = _style()
    from scipy.stats import spearmanr

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # --- panel 1: concept-shift magnitude ------------------------------------
    # Only the error column is needed; the table is ~16.8M rows.
    err = pd.read_parquet(args.concept_table, columns=["error"])["error"]
    mean_err = float(err.mean())
    n_pairs = len(err)
    if args.sample_rows and n_pairs > args.sample_rows:
        err = err.sample(args.sample_rows, random_state=0)
    ax = axes[0]
    sns.histplot(err, bins=60, color=warm, ax=ax)
    ax.set(xlabel="|measured delta|  (predicted = 0)",
           ylabel=f"trans (pert, gene) pairs (n={n_pairs:,})",
           title="Concept-shift magnitude")
    ax.axvline(mean_err, color="k", ls="--", lw=1, label=f"mean={mean_err:.3f}")
    ax.legend(frameon=False)

    # --- panel 2: mean error vs Wasserstein distance -------------------------
    ax = axes[1]
    if os.path.exists(args.summary):
        summ = pd.read_parquet(args.summary)
        if "wasserstein" in summ.columns and summ["wasserstein"].notna().any():
            s = summ.dropna(subset=["wasserstein", "mean_error"])
            r = spearmanr(s["wasserstein"], s["mean_error"]).correlation
            sns.scatterplot(data=s, x="wasserstein", y="mean_error", s=18,
                            color=cool, alpha=0.6, ax=ax, edgecolor="none")
            ax.set(xlabel="Wasserstein distance (perturbed vs control)", ylabel="mean |delta|",
                   title=f"Strength baseline  (Spearman={r:.2f})")
        else:
            ax.text(0.5, 0.5, "run step 3 for Wasserstein", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "perturbation_summary.parquet missing", ha="center", va="center")
        ax.set_axis_off()

    # --- panel 3: within-state check -----------------------------------------
    ax = axes[2]
    if os.path.exists(args.baseline):
        base = pd.read_parquet(args.baseline)
        pb = pd.read_parquet(args.pseudobulk)
        common = [g for g in base.index if g in pb.columns]
        pred = base.loc[common, args.baseline_proxy].values
        meas = pb.loc[args.ctrl, common].values
        ok = np.isfinite(pred) & np.isfinite(meas)
        rho = spearmanr(pred[ok], meas[ok]).correlation
        ax.scatter(np.log1p(pred[ok]), meas[ok], s=8, color=cool, alpha=0.4, edgecolor="none")
        ax.set(xlabel=f"log1p AG {args.baseline_proxy}", ylabel="measured control pseudobulk",
               title=f"Within-state  (Spearman={rho:.2f})")
    else:
        ax.text(0.5, 0.5, "run step 1 for AG baseline", ha="center", va="center")
        ax.set_axis_off()

    for ax in axes:
        sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"[plot] wrote {args.output}")


if __name__ == "__main__":
    main()
