"""Unit tests for concept_shift.state_shift (non-GPU, no network, no weights)."""

import numpy as np
import pandas as pd
import pytest

from concept_shift import state_shift as ss

GENES = ["g0", "g1", "g2", "g3", "g4", "g5", "g6"]


@pytest.fixture
def toy():
    """3 states + control, 7 genes.

    AG ranks g0..g6 ascending, and control expression rises with AG rank, so
    rho_ctrl == 1.0 over g0..g5. g6 is expressed nowhere (filter fodder).

      A knocks down g5 (its target)  -> rho must drop
      B knocks down g2 (its target)
      C is unchanged, target not measured
    """
    base = pd.DataFrame({"rna_pred": [1.0, 2, 3, 4, 5, 6, 7]}, index=GENES)
    pb = pd.DataFrame(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],   # control
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0],   # A: target g5 -> 0
         [0.1, 0.2, 0.0, 0.4, 0.5, 0.6, 0.0],   # B: target g2 -> 0
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]],  # C: unchanged
        index=["control", "A", "B", "C"], columns=GENES,
    )
    coords = pd.DataFrame(
        {"chr": ["chr1", "chr1", "chr2", "chr2", "chr3", "chr3", "chr4"],
         "tss": [1_000_000, 1_500_000, 1_000_000, 9_000_000, 1_000, 2_000, 5_000]},
        index=GENES,
    )
    kd = pd.DataFrame({"pert": ["A", "B", "C"], "target_var": ["g5", "g2", None]})
    return base, pb, coords, kd


def test_filter_genes_drops_never_expressed(toy):
    base, pb, _, _ = toy
    keep = ss.filter_genes(pb, base, min_states_expressed=1)   # g6 is 0 in every state
    assert "g6" not in keep
    assert set(keep) == {"g0", "g1", "g2", "g3", "g4", "g5"}


def test_filter_genes_whitelist(toy):
    base, pb, _, _ = toy
    assert ss.filter_genes(pb, base, whitelist=["g1", "g3", "not_a_gene"]) == ["g1", "g3"]


def test_filter_genes_min_mean_expr(toy):
    base, pb, _, _ = toy
    # means over A,B,C: g3=0.4, g4=0.5, g5=0.4 clear 0.35; g2=0.2 does not
    assert set(ss.filter_genes(pb, base, min_mean_expr=0.35)) == {"g3", "g4", "g5"}


def test_state_gene_masks_excludes_own_target_only(toy):
    base, pb, coords, kd = toy
    perts = ["A", "B", "C"]
    m = ss.state_gene_masks(perts, GENES, coords,
                            kd.set_index("pert")["target_var"].to_dict(), exclude_cis=False)
    gi = {g: j for j, g in enumerate(GENES)}
    assert not m[0, gi["g5"]]        # A drops its own target
    assert m[1, gi["g5"]]            # ...but g5 is still scored under B (per-state, not global)
    assert not m[1, gi["g2"]]        # B drops its own target
    assert m[2].all()                # C: target not measured -> nothing dropped


def test_state_gene_masks_cis_exclusion(toy):
    base, pb, coords, _ = toy
    m = ss.state_gene_masks(["A"], GENES, coords, {"A": "g0"},
                            exclude_cis=True, cis_window_bp=2_000_000)
    gi = {g: j for j, g in enumerate(GENES)}
    assert not m[0, gi["g1"]]        # 500 kb from g0 on chr1 -> cis
    assert m[0, gi["g2"]]            # different chromosome -> kept


def test_state_gene_masks_per_state_gene_sets(toy):
    base, pb, coords, _ = toy
    m = ss.state_gene_masks(["A", "B", "C"], GENES, coords, {"A": "g5"}, exclude_cis=False,
                            per_state_genes={"A": ["g5", "g3"], "B": ["g4"], "C": []})
    gi = {g: j for j, g in enumerate(GENES)}
    assert m[0].sum() == 1 and m[0, gi["g3"]]   # g5 requested but is A's own target
    assert m[1].sum() == 1 and m[1, gi["g4"]]
    assert m[2].sum() == 0


def _xy(base, pb, perts):
    x = np.log1p(base.loc[GENES, "rna_pred"].to_numpy(float))
    Y = pb.loc[perts, GENES].to_numpy(float)
    ctrl = pb.loc["control", GENES].to_numpy(float)
    return x, Y, ctrl


def test_state_spearman_control_matches_and_perturbed_drops(toy):
    base, pb, _, _ = toy
    perts = ["A", "B", "C"]
    x, Y, ctrl = _xy(base, pb, perts)
    m = np.ones((3, len(GENES)), dtype=bool)
    m[:, GENES.index("g6")] = False              # g6 unexpressed; exclude it
    rho_P, rho_C = ss.state_spearman(x, Y, ctrl, m)
    assert np.allclose(rho_C, 1.0)               # control ranks match AG exactly
    assert np.isclose(rho_P[2], rho_C[2])        # C unchanged -> no degradation
    assert rho_P[0] < rho_C[0]                   # A knocked down a gene -> rho drops
    assert rho_P[1] < rho_C[1]


def test_state_spearman_nan_when_too_few_genes(toy):
    base, pb, _, _ = toy
    x, Y, ctrl = _xy(base, pb, ["A"])
    m = np.zeros((1, len(GENES)), dtype=bool)
    m[0, :2] = True                              # only 2 genes -> undefined
    rho_P, rho_C = ss.state_spearman(x, Y, ctrl, m)
    assert np.isnan(rho_P[0]) and np.isnan(rho_C[0])


def test_matched_null_rho_rises_with_cells():
    """More cells -> less pseudobulk noise -> higher rho. This IS the confound."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=200)
    X = truth[None, :] + rng.normal(scale=3.0, size=(400, 200))
    nullt = ss.matched_null(X, truth, [5, 200], n_boot=8, seed=0)
    assert nullt.loc[5, "rho_null_mean"] < nullt.loc[200, "rho_null_mean"]
    assert (nullt["rho_null_sd"] > 0).all()


def test_matched_null_per_state_uses_masks():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=50)
    X = truth[None, :] + rng.normal(scale=2.0, size=(100, 50))
    masks = np.ones((2, 50), dtype=bool)
    masks[1, 25:] = False                        # second state scored on half the genes
    nullt = ss.matched_null(X, truth, [20, 20], n_boot=5, masks=masks, mode="per_state")
    assert len(nullt) == 2 and nullt["rho_null_mean"].notna().all()


def test_filter_genes_require_expressed_in_control(toy):
    base, pb, _, _ = toy
    # g6 is 0 in control; threshold 0 excludes it (strictly-greater test)
    keep = ss.filter_genes(pb, base, require_expressed_in_control=True)
    assert "g6" not in keep and set(keep) == {"g0", "g1", "g2", "g3", "g4", "g5"}
    # a positive threshold bites: control g0=0.1, g1=0.2 fall below t=0.25; g2=0.3 survives
    keep = ss.filter_genes(pb, base, expr_threshold=0.25, require_expressed_in_control=True)
    assert set(keep) == {"g2", "g3", "g4", "g5"}


def test_filter_genes_include_control_as_state(toy):
    base, pb, _, _ = toy
    # g5 is expressed in control/B/C but 0 in A. Demanding expression in ALL 3
    # perturbed states drops it; counting control too needs 4 and still drops it.
    assert "g5" not in ss.filter_genes(pb, base, min_states_expressed=3)
    assert "g5" not in ss.filter_genes(pb, base, min_states_expressed=4,
                                       include_control_as_state=True)
    # But "expressed in >=1 state" keeps it either way.
    assert "g5" in ss.filter_genes(pb, base, min_states_expressed=1)
    assert "g5" in ss.filter_genes(pb, base, min_states_expressed=1,
                                   include_control_as_state=True)
