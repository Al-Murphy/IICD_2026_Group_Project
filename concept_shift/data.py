"""
concept_shift.data
==================

Shared data-loading and filtering for the Replogle-2022 K562-essential CRISPRi
screen. This is the single source of truth for the *state* side of the
concept-shift analysis and is used by BOTH:

- the AlphaGenome baseline pipeline (``scripts/``), and
- the scFM (single-cell foundation model) embedding-displacement work.

Both consumers must operate on **exactly** the same filtered cells, pseudobulk,
and per-perturbation relevant-gene sets, so all of that logic lives here and
nowhere else.

The module is deliberately free of ``torch`` / AlphaGenome dependencies so it
imports on a light environment (``pip install concept-shift`` core deps only).

Pipeline (matches ``specs.md`` steps 1-7)
-----------------------------------------
1. ``load_replogle``        -- pertpy.data.replogle_2022_k562_essential()
2. ``detect_normalisation`` -- raw counts -> normalize_total(1e4) + log1p; else leave
3. ``filter_min_cells``     -- keep perturbations with >= 30 cells (all 8,563 genes)
4. ``pseudobulk_delta``     -- per-perturbation mean, measured delta vs control
5. ``knockdown_qc``         -- percent-of-control knockdown flag (soft; never drops)
6. ``coord_table``          -- strand-aware hg38 TSS per gene, valid chromosomes only
7. ``relevant_gene_sets``   -- per-perturbation trans DE gene set (excl. target + cis)

``prepare`` runs the whole thing and returns a :class:`ConceptShiftData` bundle,
optionally caching the filtered AnnData + tables to disk for fast reload.

Guiding principle
-----------------
AlphaGenome is state-blind: a knockdown does not change any downstream gene's
DNA sequence, so its predicted delta for every (perturbation, gene) pair is
exactly 0. The non-zero *measured* delta computed here is the concept-shift
signal. Nothing in this module ever calls AlphaGenome.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Chromosomes we keep (hg38 primary assembly, autosomes + sex).
_VALID_CHR = re.compile(r"^chr(\d+|X|Y)$")

DEFAULT_MIN_CELLS = 30
DEFAULT_CIS_WINDOW_BP = 2_000_000  # +/- 2 Mb around target TSS (KRAB spreading)
DEFAULT_PERT_COL = "perturbation"
DEFAULT_CTRL = "control"
DEFAULT_KD_PCT_THRESHOLD = -25.0  # target must drop > 25% of control to be "ok"


# ---------------------------------------------------------------------------
# Result bundle
# ---------------------------------------------------------------------------
@dataclass
class ConceptShiftData:
    """All artifacts the AlphaGenome and scFM pipelines share.

    Attributes
    ----------
    adata : AnnData
        Filtered (>= ``min_cells`` per perturbation), normalised AnnData with
        all 8,563 genes retained. Includes the ``control`` cells.
    pb : pd.DataFrame
        Pseudobulk mean expression, index = perturbation label (incl. control),
        columns = ``adata.var_names``.
    delta : pd.DataFrame
        Measured delta = ``pb - pb.loc[control]`` (same shape as ``pb``).
    ctrl_expr : pd.Series
        Control pseudobulk expression (``pb.loc[control]``).
    knockdown_qc : pd.DataFrame
        Per-perturbation soft QC flag (percent-of-control on the target gene).
    coords : pd.DataFrame
        Per-gene strand-aware hg38 coordinates (subset of ``adata.var`` on valid
        chromosomes) with an added integer ``tss`` column. Index = var_names.
    relevant_genes : dict[str, list[str]]
        ``{perturbation: [trans gene var_names]}`` -- DE vs control, excluding
        the target gene and its +/- cis-window neighbours, valid hg38 only.
    pert_to_ens : dict[str, str]
        Perturbation label -> target Ensembl gene id.
    ens_to_var : dict[str, str]
        Ensembl gene id -> ``adata`` var_name.
    pert_col, ctrl : str
        Column / control label used throughout.
    """

    adata: "AnnData"  # noqa: F821 (anndata imported lazily)
    pb: pd.DataFrame
    delta: pd.DataFrame
    ctrl_expr: pd.Series
    knockdown_qc: pd.DataFrame
    coords: pd.DataFrame
    relevant_genes: dict = field(default_factory=dict)
    pert_to_ens: dict = field(default_factory=dict)
    ens_to_var: dict = field(default_factory=dict)
    pert_col: str = DEFAULT_PERT_COL
    ctrl: str = DEFAULT_CTRL

    @property
    def perturbations(self) -> list:
        """Perturbation labels retained (control excluded)."""
        return [p for p in self.pb.index if p != self.ctrl]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_dense(x) -> np.ndarray:
    """Densify a possibly-sparse ``.X`` slice to a numpy array."""
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# ---------------------------------------------------------------------------
# Step 1 -- load
# ---------------------------------------------------------------------------
def load_replogle(cache_h5ad: Optional[str] = None):
    """Load the Replogle 2022 K562-essential screen via pertpy.

    Parameters
    ----------
    cache_h5ad : str, optional
        If given and the file exists, load from there (fast). Otherwise fetch
        via ``pertpy.data.replogle_2022_k562_essential()`` and, if a path was
        given, write it out for next time.

    Returns
    -------
    AnnData
        310,385 cells x 8,563 genes (unfiltered).
    """
    import anndata as ad

    if cache_h5ad and os.path.exists(cache_h5ad):
        print(f"[data] loading cached AnnData: {cache_h5ad}")
        return ad.read_h5ad(cache_h5ad)

    import pertpy as pt

    print("[data] fetching pertpy.data.replogle_2022_k562_essential() ...")
    adata = pt.data.replogle_2022_k562_essential()
    if cache_h5ad:
        os.makedirs(os.path.dirname(os.path.abspath(cache_h5ad)), exist_ok=True)
        print(f"[data] caching raw AnnData -> {cache_h5ad}")
        adata.write_h5ad(cache_h5ad)
    return adata


# ---------------------------------------------------------------------------
# Step 2 -- normalisation detection
# ---------------------------------------------------------------------------
def detect_normalisation(adata, n_probe: int = 50) -> bool:
    """Return True if ``adata.X`` looks like raw integer counts.

    Probes the first ``n_probe`` cells: raw == all non-negative integers.
    """
    x0 = _to_dense(adata.X[:n_probe])
    return bool(np.allclose(x0, np.round(x0)) and x0.min() >= 0)


def normalise(adata, target_sum: float = 1e4):
    """Normalise-total + log1p **in place** (only call when raw counts)."""
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def ensure_normalised(adata, target_sum: float = 1e4):
    """Normalise iff raw counts are detected; otherwise leave untouched."""
    if detect_normalisation(adata):
        print("[data] raw counts detected -> normalize_total(1e4) + log1p")
        normalise(adata, target_sum=target_sum)
    else:
        print("[data] .X already normalised (floats/negatives) -> no transform")
    return adata


# ---------------------------------------------------------------------------
# Step 3 -- 30-cell filter
# ---------------------------------------------------------------------------
def filter_min_cells(adata, min_cells: int = DEFAULT_MIN_CELLS,
                     pert_col: str = DEFAULT_PERT_COL, ctrl: str = DEFAULT_CTRL):
    """Keep perturbations (and control) with >= ``min_cells`` cells.

    Keeps ALL genes (no HVG filter -- spec is explicit about this).
    """
    vc = adata.obs[pert_col].value_counts()
    keep = vc[vc >= min_cells].index
    adata_f = adata[adata.obs[pert_col].isin(keep)].copy()
    n_pert = adata_f.obs[pert_col].nunique() - int(ctrl in set(keep))
    print(f"[data] {min_cells}-cell filter: {len(keep)} labels kept "
          f"(~{n_pert} perturbations + control), {adata_f.n_vars} genes retained")
    return adata_f


# ---------------------------------------------------------------------------
# Step 4 -- pseudobulk + measured delta
# ---------------------------------------------------------------------------
def pseudobulk_delta(adata, pert_col: str = DEFAULT_PERT_COL,
                     ctrl: str = DEFAULT_CTRL):
    """Per-perturbation mean expression and measured delta vs control.

    Returns
    -------
    pb : pd.DataFrame          (perturbation x gene) mean expression
    delta : pd.DataFrame       pb - pb.loc[ctrl]
    ctrl_expr : pd.Series      pb.loc[ctrl]
    """
    X = _to_dense(adata.X)
    pb = pd.DataFrame(
        X, index=adata.obs[pert_col].values, columns=adata.var_names
    ).groupby(level=0).mean()
    if ctrl not in pb.index:
        raise ValueError(f"control label {ctrl!r} not present after filtering")
    ctrl_expr = pb.loc[ctrl]
    delta = pb.sub(ctrl_expr, axis=1)
    return pb, delta, ctrl_expr


# ---------------------------------------------------------------------------
# Step 5 -- knockdown QC (soft flag, percent-based, never drops)
# ---------------------------------------------------------------------------
def build_target_maps(adata, pert_col: str = DEFAULT_PERT_COL):
    """Return (pert_to_ens, ens_to_var) mapping dictionaries.

    Targets are matched via ``var['ensembl_id']`` -- NEVER by symbol, because
    some var_names are ``SYMBOL_ENSG...`` style.
    """
    pert_to_ens = (
        adata.obs.drop_duplicates(pert_col)
        .set_index(pert_col)["gene_id"].astype(str).to_dict()
    )
    ens_to_var = dict(zip(adata.var["ensembl_id"].astype(str), adata.var_names))
    return pert_to_ens, ens_to_var


def knockdown_qc(pb, ctrl_expr, pert_to_ens, ens_to_var,
                 ctrl: str = DEFAULT_CTRL,
                 pct_threshold: float = DEFAULT_KD_PCT_THRESHOLD) -> pd.DataFrame:
    """Percent-of-control knockdown QC on each perturbation's own target.

    Uses percent change on the de-logged target expression, NOT logFC (logFC
    gives false "weak" calls on lowly-expressed targets). This is a SOFT flag:
    rows are carried, never dropped. Expect ~180 ``target_not_measured``.

    Returns a table with columns
    ``[pert, target_var, pct_change, status, knockdown_ok]``.
    """
    rows = []
    for P in [p for p in pb.index if p != ctrl]:
        tv = ens_to_var.get(str(pert_to_ens.get(P)))
        if tv is None:
            rows.append((P, None, np.nan, "target_not_measured"))
            continue
        c = np.expm1(ctrl_expr[tv])
        k = np.expm1(pb.loc[P, tv])
        pct = 100.0 * (k - c) / (c + 1e-9)
        rows.append((P, tv, pct, "ok"))
    kd = pd.DataFrame(rows, columns=["pert", "target_var", "pct_change", "status"])
    kd["knockdown_ok"] = kd["pct_change"] < pct_threshold
    return kd


# ---------------------------------------------------------------------------
# Step 6 -- coordinate table (strand-aware hg38 TSS)
# ---------------------------------------------------------------------------
def coord_table(adata) -> pd.DataFrame:
    """Strand-aware hg38 TSS per gene, restricted to valid chromosomes.

    TSS = ``start`` on + strand, ``end`` on - strand. Chromosome must match
    ``chr(1..22|X|Y)``. Returns a subset of ``adata.var`` (index = var_names)
    with an added integer ``tss`` column.
    """
    v = adata.var.copy()
    plus = v["strand"].astype(str).isin(["+", "1"])
    v["tss"] = np.where(plus, v["start"], v["end"]).astype(int)
    v = v[v["chr"].astype(str).str.match(_VALID_CHR)]
    return v


# ---------------------------------------------------------------------------
# Step 7 -- relevant (trans) gene set per perturbation
# ---------------------------------------------------------------------------
def _rank_genes(adata, pert_col, ctrl, method="wilcoxon"):
    """Run scanpy DE of every perturbation group vs the control reference.

    Returns a dict ``{group: DataFrame[names, logfoldchanges, pvals_adj]}``.
    This is ONE scanpy call over all groups (not per-perturbation).
    """
    import scanpy as sc

    groups = [g for g in adata.obs[pert_col].unique() if g != ctrl]
    print(f"[data] rank_genes_groups ({method}) for {len(groups)} groups vs {ctrl!r} ...")
    sc.tl.rank_genes_groups(
        adata, groupby=pert_col, groups=list(groups), reference=ctrl,
        method=method, pts=False,
    )
    res = adata.uns["rank_genes_groups"]
    out = {}
    for g in groups:
        out[g] = pd.DataFrame({
            "names": res["names"][g],
            "logfoldchanges": res["logfoldchanges"][g],
            "pvals_adj": res["pvals_adj"][g],
        })
    return out


def relevant_gene_sets(adata, coords, pert_to_ens, ens_to_var,
                       pert_col: str = DEFAULT_PERT_COL, ctrl: str = DEFAULT_CTRL,
                       cis_window_bp: int = DEFAULT_CIS_WINDOW_BP,
                       alpha: float = 0.05, min_abs_lfc: float = 0.0,
                       top_n: Optional[int] = None, method: str = "wilcoxon") -> dict:
    """Per-perturbation trans gene set for the concept-shift table.

    For each perturbation ``P`` (target ``g_P``):

    1. DE genes vs control (significant at ``pvals_adj < alpha`` AND
       ``|logfoldchanges| >= min_abs_lfc``; optionally the top ``top_n`` by
       absolute logFC).
    2. Exclude the target gene ``g_P``.
    3. Exclude *cis* neighbours: any gene on the target's chromosome whose TSS
       is within ``+/- cis_window_bp`` of the target TSS (KRAB spreading).
    4. Keep only genes with valid hg38 coordinates (present in ``coords``).

    Returns ``{P: [var_names]}``. Perturbations whose target has no valid
    coordinates keep the DE/cis logic that can still be applied (target
    excluded by id; cis skipped with a warning).
    """
    de = _rank_genes(adata, pert_col, ctrl, method=method)
    valid_genes = set(coords.index)

    relevant = {}
    n_no_target_coord = 0
    for P, df in de.items():
        sig = df[(df["pvals_adj"] < alpha) & (df["logfoldchanges"].abs() >= min_abs_lfc)]
        if top_n is not None:
            sig = sig.reindex(sig["logfoldchanges"].abs().sort_values(ascending=False).index)
            sig = sig.head(top_n)
        genes = [g for g in sig["names"].tolist() if g in valid_genes]

        # (2) exclude the target gene itself
        target_ens = str(pert_to_ens.get(P))
        target_var = ens_to_var.get(target_ens)
        genes = [g for g in genes if g != target_var]

        # (3) exclude cis neighbours around the target TSS
        if target_var is not None and target_var in coords.index:
            t_chr = str(coords.loc[target_var, "chr"])
            t_tss = int(coords.loc[target_var, "tss"])
            same_chr = coords.index[coords["chr"].astype(str) == t_chr]
            cis = set(
                coords.loc[same_chr].index[
                    (coords.loc[same_chr, "tss"].astype(int) - t_tss).abs() <= cis_window_bp
                ]
            )
            genes = [g for g in genes if g not in cis]
        else:
            n_no_target_coord += 1

        relevant[P] = genes

    if n_no_target_coord:
        warnings.warn(
            f"{n_no_target_coord} perturbations had no valid target coords; "
            "cis-neighbour exclusion skipped for those (target still excluded by id)."
        )
    total = sum(len(v) for v in relevant.values())
    print(f"[data] relevant gene sets: {len(relevant)} perturbations, "
          f"{total} (pert, gene) trans pairs total")
    return relevant


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def prepare(cache_h5ad: Optional[str] = None,
            filtered_h5ad: Optional[str] = None,
            min_cells: int = DEFAULT_MIN_CELLS,
            pert_col: str = DEFAULT_PERT_COL, ctrl: str = DEFAULT_CTRL,
            cis_window_bp: int = DEFAULT_CIS_WINDOW_BP,
            alpha: float = 0.05, min_abs_lfc: float = 0.0,
            top_n: Optional[int] = None, de_method: str = "wilcoxon",
            compute_relevant: bool = True) -> ConceptShiftData:
    """Run the full state-side pipeline (spec steps 1-7).

    Parameters
    ----------
    cache_h5ad : str, optional
        Path for the RAW pertpy download cache.
    filtered_h5ad : str, optional
        Path for the FILTERED+normalised AnnData cache. If it exists it is
        loaded directly (skipping load/normalise/filter).
    compute_relevant : bool
        If False, skip the (slow) DE step and return an empty
        ``relevant_genes`` dict -- useful for the scFM team when they only need
        the filtered cells + pseudobulk.

    Returns
    -------
    ConceptShiftData
    """
    import anndata as ad

    if filtered_h5ad and os.path.exists(filtered_h5ad):
        print(f"[data] loading filtered AnnData: {filtered_h5ad}")
        adata_f = ad.read_h5ad(filtered_h5ad)
    else:
        adata = load_replogle(cache_h5ad=cache_h5ad)
        ensure_normalised(adata)
        adata_f = filter_min_cells(adata, min_cells=min_cells,
                                   pert_col=pert_col, ctrl=ctrl)
        if filtered_h5ad:
            os.makedirs(os.path.dirname(os.path.abspath(filtered_h5ad)), exist_ok=True)
            print(f"[data] caching filtered AnnData -> {filtered_h5ad}")
            adata_f.write_h5ad(filtered_h5ad)

    pb, delta, ctrl_expr = pseudobulk_delta(adata_f, pert_col=pert_col, ctrl=ctrl)
    pert_to_ens, ens_to_var = build_target_maps(adata_f, pert_col=pert_col)
    kd = knockdown_qc(pb, ctrl_expr, pert_to_ens, ens_to_var, ctrl=ctrl)
    coords = coord_table(adata_f)

    relevant = {}
    if compute_relevant:
        relevant = relevant_gene_sets(
            adata_f, coords, pert_to_ens, ens_to_var,
            pert_col=pert_col, ctrl=ctrl, cis_window_bp=cis_window_bp,
            alpha=alpha, min_abs_lfc=min_abs_lfc, top_n=top_n, method=de_method,
        )

    return ConceptShiftData(
        adata=adata_f, pb=pb, delta=delta, ctrl_expr=ctrl_expr,
        knockdown_qc=kd, coords=coords, relevant_genes=relevant,
        pert_to_ens=pert_to_ens, ens_to_var=ens_to_var,
        pert_col=pert_col, ctrl=ctrl,
    )
