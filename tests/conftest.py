"""Shared pytest fixtures: a tiny synthetic Replogle-like AnnData.

Non-GPU, no network, no model weights. The fixture mimics the obs/var schema
that ``concept_shift.data`` relies on so the filtering / pseudobulk / relevant-
gene logic can be exercised deterministically.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_adata():
    """Small raw-count AnnData: control + 3 perturbations, 6 genes.

    Layout (genes):
      g0  chr1  1,000,000     +   <- target of pert 'A'
      g1  chr1  1,500,000     +   <- cis neighbour of g0 (within 2 Mb) -> excluded
      g2  chr1  9,000,000     +   <- trans on chr1 (> 2 Mb from g0)    -> eligible
      g3  chr2  1,000,000     -   <- target of pert 'B'
      g4  chr2  5,000,000     +   <- trans                              -> eligible
      g5  chrM  100           +   <- invalid chromosome                 -> dropped
    """
    import anndata as ad

    rng = np.random.default_rng(0)
    genes = ["g0", "g1", "g2", "g3", "g4", "g5"]
    ensembl = [f"ENSG{ i:011d}" for i in range(6)]

    var = pd.DataFrame({
        "chr": ["chr1", "chr1", "chr1", "chr2", "chr2", "chrM"],
        "start": [1_000_000, 1_500_000, 9_000_000, 1_000_000, 5_000_000, 100],
        "end":   [1_002_000, 1_502_000, 9_002_000, 1_002_000, 5_002_000, 200],
        "strand": ["+", "+", "+", "-", "+", "+"],
        "ensembl_id": ensembl,
    }, index=genes)

    # 4 groups x 40 cells each.
    labels, gene_ids = [], []
    pert_to_target = {"control": None, "A": ensembl[0], "B": ensembl[3], "C": ensembl[2]}
    for p, tgt in pert_to_target.items():
        labels += [p] * 40
        gene_ids += [str(tgt)] * 40

    n = len(labels)
    X = rng.poisson(5.0, size=(n, len(genes))).astype(np.float32)
    obs = pd.DataFrame({
        "perturbation": labels,
        "gene_id": gene_ids,
        "nperts": [0 if p == "control" else 1 for p in labels],
    }, index=[f"cell{i}" for i in range(n)])

    # Make each perturbation move its trans genes so DE has signal.
    a = np.array(labels)
    X[a == "A", 2] += 40   # A up-regulates g2 (trans, eligible)
    X[a == "A", 1] += 40   # A moves g1 too (cis -> must be excluded)
    X[a == "B", 4] += 40   # B up-regulates g4 (trans, eligible)
    X[a == "A", 0] = 0     # A knocks down its target g0
    X[a == "B", 3] = 0     # B knocks down its target g3

    return ad.AnnData(X=X, obs=obs, var=var)
