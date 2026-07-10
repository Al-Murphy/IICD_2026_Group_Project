"""
concept_shift.state_shift
=========================

Does AlphaGenome's accuracy degrade when the cell state shifts?

For each perturbed cell state ``P`` we correlate the state-blind AlphaGenome
baseline against that state's measured pseudobulk, across a gene set that
excludes the perturbed gene itself:

    rho_P    = spearman( AG, pb[P]       )   over P's gene set
    rho_ctrl = spearman( AG, pb[control] )   over the SAME genes
    d_rho    = rho_ctrl - rho_P              AG's accuracy degradation

Why Spearman, rolled up per state
---------------------------------
Spearman is invariant to any monotone transform, so AG's raw signal never has to
be calibrated onto the expression scale. That removes every units artifact:

* an isotonic/least-squares calibration fits the CONDITIONAL MEAN, so it shrinks
  (sd 0.39 vs measured 0.50) and appears to "under-predict high-expressed genes";
* absolute error in log1p space is heteroscedastic, so |error| tracks expression.

Rolling up to one number per state also avoids per-gene cancellation entirely.

The cell-count confound -- and the matched null
-----------------------------------------------
Essential-gene knockdowns kill cells: the median state holds ~125 cells against
~10,691 for control. A noisier pseudobulk correlates worse with AG *for free*
(measured: Spearman(n_cells, d_rho) = -0.46).

Correcting to the MEDIAN cell count is not good enough -- rho is steeply
nonlinear in n (0.638 at n=30, 0.666 at n=5000). We instead build a **matched
null**: subsample the CONTROL cells to each state's own cell count, ``n_boot``
times, giving the expected rho and its sd, hence a z-score per state.

    d_rho_matched = rho_null_mean(n_P) - rho_P
    z             = d_rho_matched / rho_null_sd(n_P)

Headline (full gene set, n = 1,971): rho 0.6657 -> 0.6507; d_rho raw = +0.0150
(95.4% "degrade") but d_rho matched = +0.0050 (62% degrade, only 29% at z>2).
Spearman(d_rho_matched, Wasserstein) = 0.389. AlphaGenome's gene-ranking accuracy
is almost state-invariant: 0.75% relative loss, real and strength-dependent.

Gene filtering
--------------
Two orthogonal knobs:

* **global** (:func:`filter_genes`) -- one gene list for every state, e.g. drop
  genes unexpressed in every state, or restrict to a pathway/whitelist.
* **per-state** (``per_state_genes``) -- a different gene set per perturbation,
  e.g. the regulatory network of the perturbed gene.

On top of either, each state always drops its own target gene, and optionally the
target's cis neighbours (KRAB spreading).

.. warning::
   When ``per_state_genes`` is used the gene sets differ a lot between states, so
   the matched null must be rebuilt per state (``null_mode="per_state"``).
   :func:`compute_state_shift` selects this automatically.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

CTRL = "control"

__all__ = [
    "load_inputs",
    "filter_genes",
    "state_gene_masks",
    "state_spearman",
    "control_matrix",
    "matched_null",
    "noise_floor_curve",
    "compute_state_shift",
    "plot_state_shift",
    "plot_rho_vs_cells",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_inputs(baseline="./out/ag_k562_baseline.parquet",
                pseudobulk="./out/pseudobulk.parquet",
                coords="./out/coords.parquet",
                knockdown="./out/knockdown_check.csv",
                wasserstein="./out/wasserstein_distance.parquet",
                h5ad: Optional[str] = "./out/replogle_k562_filtered.h5ad"):
    """Load every artifact the analysis needs.

    Returns a dict with keys ``base, pb, coords, kd, wass, adata`` (``adata`` is
    ``None`` when ``h5ad`` is None -- it is only needed for the matched null).
    """
    out = {
        "base": pd.read_parquet(baseline),
        "pb": pd.read_parquet(pseudobulk),
        "coords": pd.read_parquet(coords),
        "kd": pd.read_csv(knockdown),
        "wass": pd.read_parquet(wasserstein)["wasserstein"],
        "adata": None,
    }
    if h5ad:
        import anndata as ad

        out["adata"] = ad.read_h5ad(h5ad)
    return out


# ---------------------------------------------------------------------------
# Global gene filtering
# ---------------------------------------------------------------------------
def filter_genes(pb: pd.DataFrame, base: pd.DataFrame, *,
                 ctrl: str = CTRL,
                 expr_threshold: float = 0.0,
                 min_states_expressed: int = 0,
                 include_control_as_state: bool = False,
                 require_expressed_in_control: bool = False,
                 min_mean_expr: float = 0.0,
                 whitelist: Optional[Iterable[str]] = None,
                 require_coords: Optional[pd.DataFrame] = None) -> list:
    """Global gene list: genes AG scored, present in the pseudobulk, and surviving
    the requested expression / whitelist filters.

    A gene counts as "expressed" in a state when its pseudobulk **strictly
    exceeds** ``expr_threshold``. Note ``expr_threshold=0`` keeps essentially
    everything: a pseudobulk mean over hundreds of cells is rarely *exactly* 0.
    Use a small positive threshold (e.g. 0.01) for a meaningful filter.

    Parameters
    ----------
    expr_threshold, min_states_expressed
        Keep genes expressed in at least ``min_states_expressed`` states. Use
        ``min_states_expressed=1`` to drop genes unexpressed in *every* state,
        or ``len(states)`` to demand expression everywhere.
    include_control_as_state
        Count the control row as one of the states for the two filters above.
        Off by default: the perturbed states are what the analysis scores.
    require_expressed_in_control
        Keep only genes expressed in the control state. This is the natural
        filter when AG's *baseline* accuracy is the reference, since a gene that
        is silent in control contributes only noise to ``rho_ctrl``. Combines
        (AND) with the other filters.
    min_mean_expr
        Keep genes whose mean pseudobulk across states is >= this.
    whitelist
        Restrict to these genes (e.g. a regulatory network / pathway).
    require_coords
        If given, keep only genes present in this coords table (valid hg38).
    """
    genes = [g for g in pb.columns if g in base.index]
    if require_coords is not None:
        genes = [g for g in genes if g in require_coords.index]
    if whitelist is not None:
        wl = set(whitelist)
        genes = [g for g in genes if g in wl]

    rows = list(pb.index) if include_control_as_state else [i for i in pb.index if i != ctrl]
    sub = pb.loc[rows, genes]
    keep = np.ones(len(genes), dtype=bool)
    if min_states_expressed > 0:
        keep &= (sub > expr_threshold).sum(axis=0).to_numpy() >= min_states_expressed
    if require_expressed_in_control:
        keep &= pb.loc[ctrl, genes].to_numpy() > expr_threshold
    if min_mean_expr > 0:
        keep &= sub.mean(axis=0).to_numpy() >= min_mean_expr
    return [g for g, k in zip(genes, keep) if k]


# ---------------------------------------------------------------------------
# Per-state gene masks
# ---------------------------------------------------------------------------
def state_gene_masks(perts, genes, coords, target_map: Mapping[str, str], *,
                     exclude_target: bool = True,
                     exclude_cis: bool = True,
                     cis_window_bp: int = 2_000_000,
                     per_state_genes: Optional[Mapping[str, Iterable[str]]] = None) -> np.ndarray:
    """Boolean ``(n_perts, n_genes)`` mask of which genes each state is scored on.

    The exclusion is **per state, not global**: gene ``GATA1`` is dropped from the
    ``GATA1`` row only, and still scored under every other perturbation.
    """
    gpos = {g: j for j, g in enumerate(genes)}
    m = np.ones((len(perts), len(genes)), dtype=bool)

    if per_state_genes is not None:
        m[:] = False
        for i, P in enumerate(perts):
            for g in per_state_genes.get(P, ()):
                j = gpos.get(g)
                if j is not None:
                    m[i, j] = True

    have = [g for g in genes if g in coords.index]
    g_chr = pd.Series(coords.loc[have, "chr"].astype(str).to_numpy(), index=have)
    g_tss = pd.Series(coords.loc[have, "tss"].astype(np.int64).to_numpy(), index=have)
    chr_arr = np.array([g_chr.get(g, "") for g in genes], dtype=object)
    tss_arr = np.array([g_tss.get(g, -10**12) for g in genes], dtype=np.int64)

    for i, P in enumerate(perts):
        tv = target_map.get(P)
        if not isinstance(tv, str):
            continue                                  # target gene not measured
        if exclude_target:
            j = gpos.get(tv)
            if j is not None:
                m[i, j] = False
        if exclude_cis and tv in coords.index:
            tc = str(coords.loc[tv, "chr"]); tt = int(coords.loc[tv, "tss"])
            m[i] &= ~((chr_arr == tc) & (np.abs(tss_arr - tt) <= cis_window_bp))
    return m


# ---------------------------------------------------------------------------
# Per-state Spearman
# ---------------------------------------------------------------------------
def state_spearman(x: np.ndarray, Y: np.ndarray, ctrl_vec: np.ndarray,
                   masks: np.ndarray):
    """Return ``(rho_pert, rho_ctrl)`` arrays, one entry per state.

    ``rho_ctrl`` is recomputed on each state's own gene mask so the two are
    directly comparable.
    """
    n = Y.shape[0]
    rho_P = np.empty(n); rho_C = np.empty(n)
    for i in range(n):
        gm = masks[i]
        if gm.sum() < 3:
            rho_P[i] = rho_C[i] = np.nan
            continue
        rho_P[i] = spearmanr(x[gm], Y[i][gm]).statistic
        rho_C[i] = spearmanr(x[gm], ctrl_vec[gm]).statistic
    return rho_P, rho_C


# ---------------------------------------------------------------------------
# Matched null (the cell-count correction)
# ---------------------------------------------------------------------------
def control_matrix(adata, genes, ctrl: str = CTRL, pert_col: str = "perturbation"):
    """Single-cell control matrix restricted to ``genes`` (CSR if sparse)."""
    import scipy.sparse as sp

    gi = [adata.var_names.get_loc(g) for g in genes]
    ci = np.where((adata.obs[pert_col].astype(str) == ctrl).values)[0]
    X = adata.X[ci][:, gi]
    return X.tocsr() if sp.issparse(X) else X


def _submean(X, idx):
    import scipy.sparse as sp

    sub = X[idx]
    return np.asarray(sub.mean(0)).ravel() if sp.issparse(sub) else np.asarray(sub).mean(0)


def matched_null(X_ctrl, x: np.ndarray, n_values, n_boot: int = 20, *,
                 masks: Optional[np.ndarray] = None, seed: int = 0,
                 mode: str = "by_n") -> pd.DataFrame:
    """Expected rho (and its sd) for unperturbed cells at each state's cell count.

    ``mode="by_n"``
        One null per distinct cell count, scored on the full gene set. Correct
        when the per-state masks differ only by the handful of excluded target /
        cis genes.
    ``mode="per_state"``
        One null per state, scored on that state's own mask. Required when
        ``per_state_genes`` makes the gene sets genuinely different. Costs
        ``n_states * n_boot`` subsamples.

    Never match to the median cell count: rho is steeply nonlinear in n, so a
    single global match under-corrects small states and over-corrects large ones.
    """
    rng = np.random.default_rng(seed)
    n_cells_total = X_ctrl.shape[0]

    if mode == "by_n":
        rx = rankdata(x)
        rows = {}
        for nc in np.unique(np.asarray(n_values)):
            rs = []
            for _ in range(n_boot):
                idx = rng.choice(n_cells_total, min(int(nc), n_cells_total), replace=False)
                rs.append(pearsonr(rx, rankdata(_submean(X_ctrl, idx))).statistic)
            rows[int(nc)] = (float(np.mean(rs)), float(np.std(rs, ddof=1)))
        return pd.DataFrame.from_dict(
            rows, orient="index", columns=["rho_null_mean", "rho_null_sd"]
        ).rename_axis("n_cells")

    if masks is None:
        raise ValueError("mode='per_state' needs the per-state gene masks")
    mean_, sd_ = np.empty(len(n_values)), np.empty(len(n_values))
    for i, nc in enumerate(np.asarray(n_values)):
        gm = masks[i]
        if gm.sum() < 3:
            mean_[i] = sd_[i] = np.nan
            continue
        rxi = rankdata(x[gm])
        rs = []
        for _ in range(n_boot):
            idx = rng.choice(n_cells_total, min(int(nc), n_cells_total), replace=False)
            rs.append(pearsonr(rxi, rankdata(_submean(X_ctrl, idx)[gm])).statistic)
        mean_[i], sd_[i] = np.mean(rs), np.std(rs, ddof=1)
    return pd.DataFrame({"rho_null_mean": mean_, "rho_null_sd": sd_})


def noise_floor_curve(X_ctrl, x: np.ndarray, n_boot: int = 5, seed: int = 1,
                      grid=(30, 50, 100, 200, 400, 800, 1600, 3200, 5000)) -> pd.Series:
    """Smooth rho-vs-#cells curve, for the diagnostic plot."""
    rng = np.random.default_rng(seed)
    out = {}
    for nc in grid:
        rs = [spearmanr(x, _submean(X_ctrl, rng.choice(X_ctrl.shape[0],
                                                       min(nc, X_ctrl.shape[0]), replace=False))).statistic
              for _ in range(n_boot)]
        out[nc] = float(np.mean(rs))
    return pd.Series(out, name="rho_floor").rename_axis("n_cells")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def compute_state_shift(base, pb, coords, kd, wass, adata, *,
                        proxy: str = "rna_pred",
                        genes: Optional[Iterable[str]] = None,
                        per_state_genes: Optional[Mapping[str, Iterable[str]]] = None,
                        exclude_target: bool = True,
                        exclude_cis: bool = True,
                        cis_window_bp: int = 2_000_000,
                        n_boot: int = 20,
                        null_mode: Optional[str] = None,
                        seed: int = 0,
                        ctrl: str = CTRL,
                        pert_col: str = "perturbation",
                        floor_boot: int = 5):
    """Run the whole per-state analysis.

    Returns ``(res, floor)``: a per-state DataFrame and the rho-vs-#cells curve.

    ``res`` columns: ``n_cells, rho_ctrl, rho_pert, d_rho, rho_null_mean,
    rho_null_sd, d_rho_matched, z, wasserstein``.
    """
    if genes is None:
        genes = filter_genes(pb, base, ctrl=ctrl)
    genes = list(genes)
    perts = [p for p in pb.index if p != ctrl]

    x = np.log1p(base.loc[genes, proxy].to_numpy(float))   # monotone -> Spearman-safe
    Y = pb.loc[perts, genes].to_numpy(float)
    ctrl_vec = pb.loc[ctrl, genes].to_numpy(float)

    target_map = kd.set_index("pert")["target_var"].to_dict()
    masks = state_gene_masks(perts, genes, coords, target_map,
                             exclude_target=exclude_target, exclude_cis=exclude_cis,
                             cis_window_bp=cis_window_bp, per_state_genes=per_state_genes)

    rho_P, rho_C = state_spearman(x, Y, ctrl_vec, masks)

    n_cells = adata.obs[pert_col].astype(str).value_counts().reindex(perts).to_numpy()
    X_ctrl = control_matrix(adata, genes, ctrl=ctrl, pert_col=pert_col)

    if null_mode is None:
        null_mode = "per_state" if per_state_genes is not None else "by_n"
    nullt = matched_null(X_ctrl, x, n_cells, n_boot=n_boot, masks=masks,
                         seed=seed, mode=null_mode)

    res = pd.DataFrame({"pert": perts, "n_cells": n_cells,
                        "rho_ctrl": rho_C, "rho_pert": rho_P,
                        "d_rho": rho_C - rho_P}).set_index("pert")
    if null_mode == "by_n":
        res = res.join(nullt, on="n_cells")
    else:
        res["rho_null_mean"] = nullt["rho_null_mean"].to_numpy()
        res["rho_null_sd"] = nullt["rho_null_sd"].to_numpy()

    res["d_rho_matched"] = res["rho_null_mean"] - res["rho_pert"]
    res["z"] = res["d_rho_matched"] / res["rho_null_sd"]
    res = res.join(wass).dropna(subset=["wasserstein"])

    floor = noise_floor_curve(X_ctrl, x, n_boot=floor_boot, seed=seed + 1)
    res.attrs["n_genes"] = len(genes)
    res.attrs["null_mode"] = null_mode
    return res, floor


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _style():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    return plt, sns, "#d1495b", "#2e6f95", "#9aa5ad"


def plot_rho_vs_cells(res: pd.DataFrame, floor: pd.Series, ax=None):
    """The rho-vs-#cells diagnostic: perturbed states against the control noise floor.

    If the grey cloud sits *on* the red curve, the apparent degradation is just
    pseudobulk noise from having fewer cells.
    """
    plt, sns, warm, cool, grey = _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(res.n_cells, res.rho_pert, s=12, alpha=.4, color=grey,
               edgecolor="none", label="perturbed states")
    ax.plot(floor.index, floor.values, "o-", color=warm, lw=2, ms=4,
            label="noise floor (control subsampled)")
    rho = spearmanr(res.n_cells, res.d_rho).statistic
    ax.set(xscale="log", xlabel="cells in state (log)", ylabel=r"Spearman(AG, measured)",
           title=f"Confound: fewer cells $\\rightarrow$ lower $\\rho$\n"
                 f"Spearman($n$, $\\Delta\\rho$) = {rho:.2f}")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    sns.despine(ax=ax, top=True, right=True)
    return ax


def plot_state_shift(res: pd.DataFrame, floor: pd.Series):
    """Three-panel summary: rho distribution, the cell-count confound, matched-null shift."""
    plt, sns, warm, cool, grey = _style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    W = res["wasserstein"].to_numpy()

    ax = axes[0]
    sns.histplot(res.rho_pert, bins=50, color=cool, ax=ax, label="perturbed states")
    ax.axvline(res.rho_ctrl.mean(), color="k", ls="--", lw=1.2,
               label=f"control = {res.rho_ctrl.mean():.3f}")
    ax.set(xlabel=r"Spearman(AG, measured) per state", ylabel="states",
           title="AG's accuracy barely moves")
    ax.legend(frameon=False, fontsize=8)

    plot_rho_vs_cells(res, floor, ax=axes[1])

    ax = axes[2]
    sig = (res.z > 2).to_numpy()
    rho = spearmanr(res.d_rho_matched, W).statistic
    ax.scatter(W[~sig], res.d_rho_matched[~sig], s=13, alpha=.35, color=grey,
               edgecolor="none", label=r"n.s. ($z\leq2$)")
    ax.scatter(W[sig], res.d_rho_matched[sig], s=15, alpha=.65, color=cool,
               edgecolor="none", label=f"$z>2$ ({100 * sig.mean():.0f}%)")
    ax.axhline(0, color="k", lw=.8, ls="--")
    ax.set(xlabel="Wasserstein (perturbed vs control)",
           ylabel=r"$\Delta\rho$ vs matched-null control",
           title=f"Concept shift, matched-null\nSpearman = {rho:.2f}")
    ax.legend(frameon=False, fontsize=8)

    for a in axes:
        sns.despine(ax=a, top=True, right=True)
    fig.tight_layout()
    return fig
