#!/usr/bin/env python3
"""
Step 3c -- AlphaGenome error SHIFT vs perturbation strength.

This is the analysis that makes AlphaGenome's prediction load-bearing. For each
perturbation P, over its trans gene set (own target + cis excluded):

    E_P    = sum_g | AG(g) - pb[P, g] |          AG's error in the perturbed state
    E_ctrl = sum_g | AG(g) - pb[control, g] |    AG's error in the control state
    dE_P   = E_P - E_ctrl                        how much AG's error GROWS

then correlate ``dE_P`` against the per-perturbation Wasserstein distance
(n = 1,971 perturbations).

Contrast with ``build_concept_shift_table.py``, whose ``error = |measured delta|``
because ``pred_delta == 0``: there AlphaGenome cancels out entirely, so that
correlation (0.846) is between two summaries of the same expression data. Here
AG's actual predicted values enter.

Putting AG and the pseudobulk on one scale
------------------------------------------
AG's ``rna_pred`` is raw AlphaGenome signal; the pseudobulk is log1p(CP10K).

* ``--space quantile`` (DEFAULT): rank-transform AG and every expression profile
  across genes to [0,1]. Monotone, so the within-state Spearman (0.666) is
  untouched. No fit, hence no shrinkage, and errors are homoscedastic.
* ``--space expression``: fit a monotone isotonic map on the control state. This
  fits the CONDITIONAL MEAN, so it shrinks (sd 0.39 vs measured 0.50) and
  "under-predicts high-expressed genes" -- an artifact of the fit, not of AG. It
  also makes |r| scale with expression (corr +0.33), which biases gene selection
  toward low-expressed genes. In quantile space that correlation is -0.11.

No-leakage design: control cells are split in half. Calibration and gene
selection use half A; dE is scored against held-out half B, so the residual r
and the delta D share no control-estimate noise.

Results (n = 1,971)
-------------------
                          space=expression    space=quantile
  dE, all genes                  -0.103            +0.480
  dE, keep_pct=10                +0.803            +0.746
  corr(dE, sum|D|) @ 10%          0.999             0.999

Writing pred = ctrl + r and D = pb_P - ctrl:

    dE(L1) = sum(|r - D| - |r|)   ~ -sum(sign(r)*D)   when |r| >> |D|  (cancels)
                                  ->  sum|D| - sum|r| when |r| << |D|  (magnitude)

So filtering to genes AG predicts well flips the sign -- not a selection artifact
(random subsets of equal size stay ~ -0.09). BUT corr(dE, sum|D|) -> 1.000: as
|r| -> 0 the prediction converges to the control profile and dE degenerates to
the AG-free magnitude. Because predicted delta == 0, AG enters only as a per-gene
constant offset and carries no information about D; its accuracy governs how much
it CORRUPTS the measured signal, never how much it adds. This holds in both spaces.

Outputs: out/ag_error_shift.parquet  (per-perturbation dE, components, Wasserstein)

Usage
-----
    python scripts/3_controls/ag_error_shift.py                       # quantile, all genes
    python scripts/3_controls/ag_error_shift.py --keep_pct 10         # the spec's clause
    python scripts/3_controls/ag_error_shift.py --space expression    # old default
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

CTRL = "control"


def parse_args():
    p = argparse.ArgumentParser(
        description="AlphaGenome error shift (E_perturbed - E_control) vs Wasserstein.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline", default="./out/ag_k562_baseline.parquet")
    p.add_argument("--pseudobulk", default="./out/pseudobulk.parquet")
    p.add_argument("--coords", default="./out/coords.parquet")
    p.add_argument("--knockdown_csv", default="./out/knockdown_check.csv")
    p.add_argument("--wasserstein", default="./out/wasserstein_distance.parquet")
    p.add_argument("--proxy", default="rna_pred", choices=["rna_pred", "cage_pred"])
    p.add_argument("--space", default="quantile", choices=["quantile", "expression"],
                   help="quantile: rank-transform AG and every expression profile to [0,1] "
                        "-- no calibration fit, no shrinkage, homoscedastic, and AG's accuracy "
                        "stops correlating with expression level (corr 0.33 -> -0.11). "
                        "expression: log1p(CP10K) units, needs --calibration.")
    p.add_argument("--calibration", default="isotonic", choices=["isotonic", "linear", "none"],
                   help="Only used with --space expression. Isotonic fits the conditional mean, "
                        "so it SHRINKS (sd 0.39 vs 0.50) and under-predicts high-expressed genes "
                        "by construction -- an artifact of the fit, not of AlphaGenome.")
    p.add_argument("--split_halves", default="./out/control_split_halves.parquet",
                   help="Control-cell split halves. Calibration + gene selection use half A; "
                        "dE is evaluated against held-out half B, so r and D share no noise. "
                        "Falls back to the full control row if absent.")
    p.add_argument("--keep_pct", type=float, default=100.0,
                   help="Keep only genes in the bottom X%% of in-state |residual| -- i.e. genes "
                        "AlphaGenome predicts well in the control state (the spec's clause). "
                        "100 = no filtering.")
    p.add_argument("--no_exclude_cis", dest="exclude_cis", action="store_false",
                   help="Keep cis neighbours (default: excluded, +/- cis_window_bp).")
    p.add_argument("--cis_window_bp", type=int, default=2_000_000)
    p.add_argument("--out", default="./out/ag_error_shift.parquet")
    return p.parse_args()


def calibrate(x, y, how):
    """Map AG signal x onto measured scale y, fitted on the CONTROL state."""
    if how == "none":
        return x
    if how == "linear":
        b, a = np.polyfit(x, y, 1)
        return a + b * x
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip").fit(x, y).predict(x)


def trans_mask(perts, genes, coords, t2v, exclude_cis, cis_bp):
    """Per-perturbation mask: drop own target (+ optionally its cis neighbours)."""
    gpos = {g: j for j, g in enumerate(genes)}
    g_chr = coords.loc[genes, "chr"].astype(str).to_numpy()
    g_tss = coords.loc[genes, "tss"].astype(np.int64).to_numpy()
    m = np.ones((len(perts), len(genes)), dtype=bool)
    for i, P in enumerate(perts):
        tv = t2v.get(P)
        if not isinstance(tv, str):
            continue                       # target_not_measured
        j = gpos.get(tv)
        if j is not None:
            m[i, j] = False
        if exclude_cis and tv in coords.index:
            tc, tt = str(coords.loc[tv, "chr"]), int(coords.loc[tv, "tss"])
            m[i] &= ~((g_chr == tc) & (np.abs(g_tss - tt) <= cis_bp))
    return m


def main():
    args = parse_args()
    base = pd.read_parquet(args.baseline)
    pb = pd.read_parquet(args.pseudobulk)
    coords = pd.read_parquet(args.coords)
    kd = pd.read_csv(args.knockdown_csv)
    w = pd.read_parquet(args.wasserstein)["wasserstein"]

    # Split-half control: fit + select on A, evaluate against held-out B, so the
    # calibration residual r and the measured delta D share no control-estimate noise.
    halves = None
    if args.split_halves and os.path.exists(args.split_halves):
        halves = pd.read_parquet(args.split_halves)

    if halves is not None:
        genes = [g for g in pb.columns if g in base.index and g in halves.index]
        y_fit = halves.loc[genes, "ctrl_A"].to_numpy(float)   # calibration + gene selection
        y_ref = halves.loc[genes, "ctrl_B"].to_numpy(float)   # held-out control reference
    else:
        genes = [g for g in pb.columns if g in base.index]
        y_fit = y_ref = pb.loc[CTRL, genes].to_numpy(float)
    perts = [p for p in pb.index if p != CTRL]
    x = np.log1p(base.loc[genes, args.proxy].to_numpy(float))
    Y = pb.loc[perts, genes].to_numpy(float)

    if args.space == "quantile":
        # Rank-transform every profile across genes -> [0,1]. Monotone, so the
        # within-state Spearman is untouched, but scale/heteroscedasticity cannot
        # drive gene selection, and no calibration fit (hence no shrinkage) is needed.
        from scipy.stats import rankdata

        n = len(genes)
        pred = rankdata(x) / n
        r = pred - rankdata(y_fit) / n
        D = np.vstack([rankdata(Y[i]) / n for i in range(len(perts))]) - (rankdata(y_ref) / n)[None, :]
    else:
        pred = calibrate(x, y_fit, args.calibration)
        r = pred - y_fit                           # within-state residual (from half A)
        D = Y - y_ref[None, :]                     # measured delta vs held-out control

    m = trans_mask(perts, genes, coords, kd.set_index("pert")["target_var"].to_dict(),
                   args.exclude_cis, args.cis_window_bp)

    # "genes AlphaGenome predicts acceptably in-state": bottom keep_pct of |r|.
    if args.keep_pct < 100.0:
        thr = np.percentile(np.abs(r), args.keep_pct)
        m &= (np.abs(r) <= thr)[None, :]
    n_g = m.sum(1)

    E_ctrl = np.where(m, np.abs(r[None, :]), 0.0).sum(1)
    E_pert = np.where(m, np.abs(r[None, :] - D), 0.0).sum(1)
    dE_L1 = E_pert - E_ctrl
    dE_L2 = np.where(m, (r[None, :] - D) ** 2 - r[None, :] ** 2, 0.0).sum(1)
    mag = np.where(m, D ** 2, 0.0).sum(1)          # magnitude component of dE_L2
    directional = np.where(m, -2.0 * r[None, :] * D, 0.0).sum(1)

    res = pd.DataFrame({
        "pert": perts, "n_genes": n_g,
        "E_control": E_ctrl, "E_perturbed": E_pert,
        "dE_sum": dE_L1, "dE_mean": dE_L1 / n_g,
        "dE_sq": dE_L2, "magnitude_sumD2": mag, "directional_term": directional,
        "sum_abs_delta": np.where(m, np.abs(D), 0.0).sum(1),
    }).set_index("pert").join(w).dropna(subset=["wasserstein"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_parquet(args.out)

    W = res["wasserstein"].to_numpy()
    sel = np.abs(r)[m.any(0)]
    print(f"proxy={args.proxy}  space={args.space}  keep_pct={args.keep_pct}  "
          f"split_half_control={halves is not None}")
    print(f"n_perturbations={len(res)}  genes/pert={n_g.min()}-{n_g.max()}")
    print(f"mean |residual r| = {sel.mean():.4f}   mean |delta| = {np.abs(D[m]).mean():.4f}"
          f"   ratio = {sel.mean() / np.abs(D[m]).mean():.2f}x\n")
    print("Spearman vs Wasserstein (n=%d):" % len(res))
    for col in ["dE_sum", "dE_sq", "magnitude_sumD2", "directional_term", "sum_abs_delta"]:
        rho = spearmanr(res[col], W).statistic
        pr = pearsonr(res[col], W).statistic
        print(f"  {col:<18} Spearman={rho: .3f}   Pearson={pr: .3f}")

    # Degeneracy check: as |r| -> 0, pred -> control and dE -> sum|D| (AG drops out).
    deg = spearmanr(res["dE_sum"], res["sum_abs_delta"]).statistic
    print(f"\ncorr(dE_sum, sum_abs_delta) = {deg:.3f}"
          f"   <- ~1.0 means dE has degenerated to the AG-free magnitude")
    print(f"fraction of perturbations with dE_sum > 0: {100 * (res['dE_sum'] > 0).mean():.1f}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
