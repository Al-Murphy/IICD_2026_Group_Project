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
# scPerturb-hosted Replogle 2022 K562 essential screen (1.44 GB).
# NOTE: do NOT use pertpy.dt.replogle_2022_k562_essential() -- in pertpy 1.0.3
# that function is bugged and downloads the Gasperini 2019 at-scale file instead
# (verified in its source). We fetch the canonical scPerturb file directly.
REPLOGLE_ESSENTIAL_URL = (
    "https://zenodo.org/records/13350497/files/"
    "ReplogleWeissman2022_K562_essential.h5ad?download=1"
)


def load_replogle(cache_h5ad: Optional[str] = None):
    """Load the Replogle 2022 K562-essential screen (scPerturb).

    Parameters
    ----------
    cache_h5ad : str, optional
        If given and present, load from there (fast). Otherwise download the
        canonical scPerturb h5ad to that path and load it.

    Returns
    -------
    AnnData
        ~310k cells x ~8.5k genes (unfiltered), clean single-gene CRISPRi screen.
    """
    import anndata as ad

    if cache_h5ad and os.path.exists(cache_h5ad):
        print(f"[data] loading cached AnnData: {cache_h5ad}")
        return ad.read_h5ad(cache_h5ad)

    import urllib.request

    dest = cache_h5ad or "./ReplogleWeissman2022_K562_essential.h5ad"
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    print(f"[data] downloading Replogle K562-essential (scPerturb, 1.44 GB) -> {dest}")
    urllib.request.urlretrieve(REPLOGLE_ESSENTIAL_URL, dest)
    return ad.read_h5ad(dest)


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
# Orchestrator
# ---------------------------------------------------------------------------
def prepare(cache_h5ad: Optional[str] = None,
            filtered_h5ad: Optional[str] = None,
            min_cells: int = DEFAULT_MIN_CELLS,
            pert_col: str = DEFAULT_PERT_COL,
            ctrl: str = DEFAULT_CTRL) -> ConceptShiftData:
    """Run the full state-side pipeline (load -> filter -> pseudobulk -> QC -> coords).

    Parameters
    ----------
    cache_h5ad : str, optional
        Path for the RAW scPerturb download cache.
    filtered_h5ad : str, optional
        Path for the FILTERED+normalised AnnData cache. If it exists it is
        loaded directly (skipping load/normalise/filter).

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


    return ConceptShiftData(
        adata=adata_f, pb=pb, delta=delta, ctrl_expr=ctrl_expr,
        knockdown_qc=kd, coords=coords,
        pert_to_ens=pert_to_ens, ens_to_var=ens_to_var,
        pert_col=pert_col, ctrl=ctrl,
    )
