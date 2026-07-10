#!/usr/bin/env python3
"""
Step 3d -- AG's correlation with each cell state (the simple, rolled-up question).

For each of the 1,971 perturbed cell states, correlate the AlphaGenome baseline
against that state's measured pseudobulk across its **trans** genes (own target +
cis excluded), and compare to the control:

    rho_P    = spearman( AG, pb[P]       )   over P's trans genes
    rho_ctrl = spearman( AG, pb[control] )   over the SAME genes
    d_rho    = rho_ctrl - rho_P              how much AG's accuracy DEGRADES

Why this is the right shape of question
--------------------------------------
Spearman is invariant to any monotone transform, so AG's raw signal needs no
calibration onto the expression scale -- no isotonic fit, no shrinkage, no
heteroscedasticity, none of the units artifacts that plague the per-gene error
shift (``ag_error_shift.py``). Rolling up to one number per state also avoids the
per-gene |r| >> |D| cancellation entirely.

The cell-count confound (this is the whole ballgame)
----------------------------------------------------
Essential-gene knockdowns kill cells: the median state has only 125 cells vs
10,691 for control. A noisier pseudobulk correlates worse with AG *for free*.
Measured noise floor (subsampling control cells):

    n=30 -> rho 0.638      n=200 -> rho 0.660     n=5000 -> rho 0.6656

Raw d_rho is therefore mostly noise. Correcting to the MEDIAN cell count would
leave a big residual confound (rho is steeply nonlinear in n), so we build a
**matched null**: subsample the control to each state's EXACT cell count, B times,
giving the expected rho and its sd -> a z-score per state.

    mean d_rho raw     = +0.0150   (95.4% of states "degrade")
    mean d_rho MATCHED = +0.0050   (62.0% degrade; only 29.1% at z>2)

    Spearman(d_rho raw,     Wasserstein) = 0.450
    Spearman(d_rho MATCHED, Wasserstein) = 0.389

The Wasserstein relationship survives because n_cells barely tracks Wasserstein
(-0.075). Residual depth confound is small: median ncounts per state correlates
-0.089 with d_rho_matched (though -0.381 with Wasserstein).

Interpretation: AlphaGenome's gene-ranking accuracy is *almost* state-invariant.
Perturbations move cells a long way (median Wasserstein 105) yet cost AG only
~0.005 of Spearman on a base of 0.666 -- 0.75% relative, and only ~29% of states
degrade beyond matched noise. That is the concept shift, honestly sized.

Outputs: out/ag_state_correlation.parquet, out/rho_noise_floor.parquet,
         results/plots/ag_state_correlation.png

Usage
-----
    python scripts/3_controls/ag_state_correlation.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

CTRL = "control"


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-cell-state Spearman of AG vs measured expression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline", default="./out/ag_k562_baseline.parquet")
    p.add_argument("--pseudobulk", default="./out/pseudobulk.parquet")
    p.add_argument("--coords", default="./out/coords.parquet")
    p.add_argument("--knockdown_csv", default="./out/knockdown_check.csv")
    p.add_argument("--wasserstein", default="./out/wasserstein_distance.parquet")
    p.add_argument("--h5ad", default="./out/replogle_k562_filtered.h5ad",
                   help="Needed for cell counts + the subsampling noise floor.")
    p.add_argument("--proxy", default="rna_pred", choices=["rna_pred", "cage_pred"])
    p.add_argument("--no_exclude_cis", dest="exclude_cis", action="store_false")
    p.add_argument("--cis_window_bp", type=int, default=2_000_000)
    p.add_argument("--n_boot", type=int, default=20,
                   help="Control subsamples per distinct cell count (matched null).")
    p.add_argument("--out", default="./out/ag_state_correlation.parquet")
    p.add_argument("--plot", default="./results/plots/ag_state_correlation.png")
    return p.parse_args()


def trans_mask(perts, genes, coords, t2v, exclude_cis, cis_bp):
    gpos = {g: j for j, g in enumerate(genes)}
    g_chr = coords.loc[genes, "chr"].astype(str).to_numpy()
    g_tss = coords.loc[genes, "tss"].astype(np.int64).to_numpy()
    m = np.ones((len(perts), len(genes)), dtype=bool)
    for i, P in enumerate(perts):
        tv = t2v.get(P)
        if not isinstance(tv, str):
            continue
        j = gpos.get(tv)
        if j is not None:
            m[i, j] = False
        if exclude_cis and tv in coords.index:
            tc, tt = str(coords.loc[tv, "chr"]), int(coords.loc[tv, "tss"])
            m[i] &= ~((g_chr == tc) & (np.abs(g_tss - tt) <= cis_bp))
    return m


def _control_matrix(adata, genes):
    import scipy.sparse as sp

    gi = [adata.var_names.get_loc(g) for g in genes]
    ci = np.where((adata.obs["perturbation"].astype(str) == CTRL).values)[0]
    X = adata.X[ci][:, gi]
    return X.tocsr() if sp.issparse(X) else X


def matched_null(adata, genes, x, n_values, n_boot, seed=0):
    """Matched null: subsample CONTROL cells to each state's EXACT cell count.

    Matching to the median instead would leave a large residual confound -- rho is
    steeply nonlinear in n over the observed 30..1996 range (0.638 -> 0.665). We
    therefore build the null per distinct n, giving both the expected rho and its
    sd (hence a z-score per state).

    Returns a DataFrame indexed by n with columns [rho_null_mean, rho_null_sd].
    """
    import scipy.sparse as sp
    from scipy.stats import rankdata

    X = _control_matrix(adata, genes)
    rx = rankdata(x)
    rng = np.random.default_rng(seed)
    rows = {}
    for nc in np.unique(n_values):
        rs = []
        for _ in range(n_boot):
            s = rng.choice(X.shape[0], min(int(nc), X.shape[0]), replace=False)
            sub = X[s]
            mu = np.asarray(sub.mean(0)).ravel() if sp.issparse(sub) else np.asarray(sub).mean(0)
            rs.append(pearsonr(rx, rankdata(mu)).statistic)   # == Spearman(x, mu)
        rows[int(nc)] = (float(np.mean(rs)), float(np.std(rs, ddof=1)))
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=["rho_null_mean", "rho_null_sd"]).rename_axis("n_cells")


def floor_curve(adata, genes, x, n_boot, seed=1):
    """Smooth rho-vs-#cells curve, for plotting only."""
    import scipy.sparse as sp

    X = _control_matrix(adata, genes)
    rng = np.random.default_rng(seed)
    out = {}
    for nc in [30, 50, 100, 200, 400, 800, 1600, 3200, 5000]:
        rs = []
        for _ in range(n_boot):
            s = rng.choice(X.shape[0], min(nc, X.shape[0]), replace=False)
            sub = X[s]
            mu = np.asarray(sub.mean(0)).ravel() if sp.issparse(sub) else np.asarray(sub).mean(0)
            rs.append(spearmanr(x, mu).statistic)
        out[nc] = float(np.mean(rs))
    return pd.Series(out, name="rho_floor").rename_axis("n_cells")


def main():
    args = parse_args()
    import anndata as ad

    base = pd.read_parquet(args.baseline)
    pb = pd.read_parquet(args.pseudobulk)
    coords = pd.read_parquet(args.coords)
    kd = pd.read_csv(args.knockdown_csv)
    w = pd.read_parquet(args.wasserstein)["wasserstein"]
    adata = ad.read_h5ad(args.h5ad)

    genes = [g for g in pb.columns if g in base.index]
    perts = [p for p in pb.index if p != CTRL]
    x = np.log1p(base.loc[genes, args.proxy].to_numpy(float))  # Spearman: monotone-invariant
    Y = pb.loc[perts, genes].to_numpy(float)
    ctrl = pb.loc[CTRL, genes].to_numpy(float)

    m = trans_mask(perts, genes, coords, kd.set_index("pert")["target_var"].to_dict(),
                   args.exclude_cis, args.cis_window_bp)

    rho_P = np.empty(len(perts)); rho_C = np.empty(len(perts))
    for i in range(len(perts)):
        gm = m[i]
        rho_P[i] = spearmanr(x[gm], Y[i][gm]).statistic
        rho_C[i] = spearmanr(x[gm], ctrl[gm]).statistic     # same genes -> comparable

    cnt = adata.obs["perturbation"].astype(str).value_counts()
    n_cells = cnt.reindex(perts).to_numpy()

    print(f"building matched null: {len(np.unique(n_cells))} distinct cell counts "
          f"x {args.n_boot} subsamples ...")
    nullt = matched_null(adata, genes, x, n_cells, args.n_boot)
    floor = floor_curve(adata, genes, x, min(args.n_boot, 5))
    floor.to_frame().to_parquet("./out/rho_noise_floor.parquet")

    res = pd.DataFrame({
        "pert": perts, "n_cells": n_cells,
        "rho_ctrl": rho_C, "rho_pert": rho_P, "d_rho": rho_C - rho_P,
    }).set_index("pert").join(w).dropna(subset=["wasserstein"])
    res = res.join(nullt, on="n_cells")
    res["d_rho_matched"] = res["rho_null_mean"] - res["rho_pert"]   # degradation beyond matched noise
    res["z"] = res["d_rho_matched"] / res["rho_null_sd"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_parquet(args.out)

    W = res["wasserstein"].to_numpy()
    print(f"\nAG vs measured, per cell state (Spearman over trans genes), n={len(res)}\n")
    print(f"  rho control   = {res.rho_ctrl.mean():.4f}")
    print(f"  rho perturbed = {res.rho_pert.mean():.4f}")
    print(f"  cells/state: min={res.n_cells.min()} median={int(res.n_cells.median())} max={res.n_cells.max()}"
          f"   (control = {int(cnt[CTRL])})")
    print(f"\n  CONFOUND  Spearman(n_cells, d_rho)       = {spearmanr(res.n_cells, res.d_rho).statistic: .3f}"
          f"   <- fewer cells -> lower rho for free")
    print(f"            Spearman(n_cells, wasserstein) = {spearmanr(res.n_cells, W).statistic: .3f}"
          f"   <- but n_cells barely tracks strength")
    print(f"\n  mean d_rho raw     = {res.d_rho.mean():+.4f}  ({100*(res.d_rho>0).mean():.1f}% degrade)")
    print(f"  mean d_rho MATCHED = {res.d_rho_matched.mean():+.4f}  "
          f"({100*(res.d_rho_matched>0).mean():.1f}% degrade)   <- most of the raw drop was noise")
    print(f"  states with z > 2  : {100*(res.z>2).mean():.1f}%       z > 3: {100*(res.z>3).mean():.1f}%")
    print(f"\n  Spearman(d_rho raw,     W) = {spearmanr(res.d_rho, W).statistic: .3f}")
    print(f"  Spearman(d_rho MATCHED, W) = {spearmanr(res.d_rho_matched, W).statistic: .3f}")
    print(f"  Spearman(z,             W) = {spearmanr(res.z, W).statistic: .3f}")
    print(f"\n  effect size: {res.d_rho_matched.mean():.4f} on a base of {res.rho_ctrl.mean():.3f}"
          f"  = {100*res.d_rho_matched.mean()/res.rho_ctrl.mean():.2f}% relative")

    # ---------------- plot ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    warm, cool, grey = "#d1495b", "#2e6f95", "#9aa5ad"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    sns.histplot(res.rho_pert, bins=50, color=cool, ax=ax, label="perturbed states")
    ax.axvline(res.rho_ctrl.mean(), color="k", ls="--", lw=1.2,
               label=f"control = {res.rho_ctrl.mean():.3f}")
    ax.set(xlabel=r"Spearman(AG, measured) per state", ylabel="states",
           title="AG's accuracy barely moves")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.scatter(res.n_cells, res.rho_pert, s=12, alpha=.45, color=grey,
               edgecolor="none", label="perturbed states")
    ax.plot(floor.index, floor.values, "o-", color=warm, lw=2, ms=4,
            label="noise floor (control subsampled)")
    ax.set(xscale="log", xlabel="cells in state (log)", ylabel=r"Spearman(AG, measured)",
           title=f"Confound: fewer cells -> lower rho\nSpearman(n, $\\Delta\\rho$) = "
                 f"{spearmanr(res.n_cells, res.d_rho).statistic:.2f}")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[2]
    rho = spearmanr(res.d_rho_matched, W).statistic
    sig = res.z > 2
    ax.scatter(W[~sig.values], res.d_rho_matched[~sig], s=13, alpha=.35, color=grey,
               edgecolor="none", label="n.s. (z<=2)")
    ax.scatter(W[sig.values], res.d_rho_matched[sig], s=15, alpha=.65, color=cool,
               edgecolor="none", label=f"z>2 ({100*sig.mean():.0f}%)")
    ax.axhline(0, color="k", lw=.8, ls="--")
    ax.set(xlabel="Wasserstein (perturbed vs control)",
           ylabel=r"$\Delta\rho$ vs matched-null control",
           title=f"Concept shift, matched-null\nSpearman = {rho:.2f}")
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        sns.despine(ax=a, top=True, right=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    fig.savefig(args.plot, dpi=200, bbox_inches="tight")
    print(f"\nwrote {args.out} and {args.plot}")


if __name__ == "__main__":
    main()
