"""Unit tests for concept_shift.data (spec steps 1-7)."""

import numpy as np
import pandas as pd

from concept_shift import data


def test_detect_normalisation_raw(tiny_adata):
    assert data.detect_normalisation(tiny_adata) is True


def test_detect_normalisation_after_transform(tiny_adata):
    data.ensure_normalised(tiny_adata)
    # Now floats in [0, log(1e4+1)] -> not raw counts anymore.
    assert data.detect_normalisation(tiny_adata) is False


def test_filter_min_cells_keeps_all_and_control(tiny_adata):
    af = data.filter_min_cells(tiny_adata, min_cells=30)
    # All four groups have 40 cells >= 30.
    assert set(af.obs["perturbation"].unique()) == {"control", "A", "B", "C"}
    assert af.n_vars == tiny_adata.n_vars  # no HVG filter


def test_filter_min_cells_drops_small_group(tiny_adata):
    # Drop group C down to 10 cells; it should be filtered out.
    mask = ~((tiny_adata.obs["perturbation"] == "C").values
             & (np.arange(tiny_adata.n_obs) % 4 != 0))
    sub = tiny_adata[mask].copy()
    af = data.filter_min_cells(sub, min_cells=30)
    assert "C" not in set(af.obs["perturbation"].unique())


def test_pseudobulk_delta_control_is_zero(tiny_adata):
    data.ensure_normalised(tiny_adata)
    pb, delta, ctrl_expr = data.pseudobulk_delta(tiny_adata)
    assert np.allclose(delta.loc["control"].values, 0.0)
    assert list(pb.columns) == list(tiny_adata.var_names)


def test_knockdown_qc_flags_target(tiny_adata):
    data.ensure_normalised(tiny_adata)
    pb, delta, ctrl_expr = data.pseudobulk_delta(tiny_adata)
    p2e, e2v = data.build_target_maps(tiny_adata)
    kd = data.knockdown_qc(pb, ctrl_expr, p2e, e2v)
    # A and B knocked down their targets to 0 -> pct_change ~ -100, flagged ok.
    a = kd.set_index("pert")
    assert a.loc["A", "knockdown_ok"]
    assert a.loc["B", "knockdown_ok"]
    assert (a["pct_change"] < 0).any()


def test_coord_table_strand_and_valid_chr(tiny_adata):
    coords = data.coord_table(tiny_adata)
    assert "g5" not in coords.index          # chrM dropped
    assert coords.loc["g0", "tss"] == 1_000_000   # + strand -> start
    assert coords.loc["g3", "tss"] == 1_002_000   # - strand -> end


def test_relevant_gene_sets_excludes_target_and_cis(tiny_adata):
    data.ensure_normalised(tiny_adata)
    coords = data.coord_table(tiny_adata)
    p2e, e2v = data.build_target_maps(tiny_adata)
    rel = data.relevant_gene_sets(
        tiny_adata, coords, p2e, e2v, alpha=0.5, cis_window_bp=2_000_000
    )
    # Pert A: target g0 excluded; g1 is cis (<2Mb) excluded; g2 trans retained.
    assert "g0" not in rel["A"]
    assert "g1" not in rel["A"]
    assert "g2" in rel["A"]
    # No invalid-chr gene ever appears.
    assert all("g5" not in genes for genes in rel.values())
