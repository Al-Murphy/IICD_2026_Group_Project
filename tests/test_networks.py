"""Unit tests for concept_shift.networks (no network access; synthetic edges)."""

import numpy as np
import pandas as pd
import pytest

from concept_shift import networks as nw


@pytest.fixture
def edges():
    #  A -- B (900),  A -- C (700),  A -- D (500),  B -- C (800),  E -- F (750)
    return pd.DataFrame(
        {"a": ["A", "A", "A", "B", "E"],
         "b": ["B", "C", "D", "C", "F"],
         "score": np.array([900, 700, 500, 800, 750], dtype=np.int16)}
    )


def test_string_neighbours_score_cutoff(edges):
    nb = nw.string_neighbours(edges, min_score=700)
    assert nb["A"] == {"B", "C"}          # A--D (500) is below cutoff
    assert nb["B"] == {"A", "C"}
    assert nb["E"] == {"F"}
    nb900 = nw.string_neighbours(edges, min_score=900)
    assert nb900["A"] == {"B"}


def test_string_neighbours_restrict_to_measured(edges):
    """Both ends must be measured -- this is what makes it the K562-active subnetwork."""
    nb = nw.string_neighbours(edges, min_score=700, restrict_to={"A", "B", "C"})
    assert nb["A"] == {"B", "C"}
    assert "E" not in nb and "F" not in nb


def test_per_state_genes_min_genes_filter(edges):
    nb = nw.string_neighbours(edges, min_score=700)
    target_map = {"pA": "A", "pE": "E", "pMissing": None}
    measured = ["A", "B", "C", "D", "E", "F"]
    out = nw.per_state_genes_from_network(nb, target_map, measured, min_genes=2)
    assert set(out) == {"pA"}                     # E has only 1 neighbour; pMissing unmapped
    assert set(out["pA"]) == {"B", "C"}
    out1 = nw.per_state_genes_from_network(nb, target_map, measured, min_genes=1)
    assert set(out1) == {"pA", "pE"}


def test_string_neighbours_drops_self_loops(edges):
    e = pd.concat([edges, pd.DataFrame({"a": ["A"], "b": ["A"],
                                        "score": np.array([900], dtype=np.int16)})])
    nb = nw.string_neighbours(e, min_score=700)
    assert "A" not in nb.get("A", set())          # a gene is never its own neighbour


def test_per_state_genes_keeps_target_for_state_gene_masks(edges):
    """networks.py must NOT drop the target -- state_gene_masks owns that exclusion."""
    nb = nw.string_neighbours(edges, min_score=700)
    nb["A"] = nb["A"] | {"A"}                     # pretend the target crept in
    out = nw.per_state_genes_from_network(nb, {"pA": "A"}, ["A", "B", "C"], min_genes=1)
    assert "A" in out["pA"]                       # left in place, removed downstream


def test_per_state_genes_max_genes_takes_top_scoring(edges):
    nb = nw.string_neighbours(edges, min_score=500)
    out = nw.per_state_genes_from_network(
        nb, {"pA": "A"}, ["A", "B", "C", "D"], min_genes=1, max_genes=2,
        edges=edges, min_score=500,
    )
    assert set(out["pA"]) == {"B", "C"}           # 900 and 800/700 beat D's 500


def test_matched_background_matches_size_and_expression():
    genes = [f"g{i}" for i in range(100)]
    pb = pd.DataFrame([np.linspace(0.01, 5.0, 100)], index=["control"], columns=genes)
    net = {"s1": genes[:5]}                        # 5 of the lowest-expressed decile
    bg = nw.matched_background(net, pb, genes, n_bins=10, seed=0)
    assert len(bg["s1"]) == 5                      # size-matched
    assert not set(bg["s1"]) & set(net["s1"])      # disjoint from the network
    assert pb.loc["control", bg["s1"]].max() <= pb.loc["control", genes[:10]].max() * 1.01


def test_matched_background_size_matched_when_decile_exhausted():
    """Network swallows a whole decile -> must top up, not silently return fewer."""
    genes = [f"g{i}" for i in range(100)]
    pb = pd.DataFrame([np.linspace(0.01, 5.0, 100)], index=["control"], columns=genes)
    net = {"s1": genes[:10]}                       # the ENTIRE lowest decile
    bg = nw.matched_background(net, pb, genes, n_bins=10, seed=0)
    assert len(bg["s1"]) == 10                     # still size-matched
    assert not set(bg["s1"]) & set(net["s1"])


def test_gwps_downstream_sets_picks_top_absolute_z():
    z = pd.DataFrame(
        [[0.0, 3.0, -4.0, 0.5, 0.1]],                        # target A's response profile
        index=["A"], columns=["A", "B", "C", "D", "E"],
    )
    out = nw.gwps_downstream_sets(z, {"pA": "A", "pMiss": "ZZZ"},
                                  ["A", "B", "C", "D", "E"], top_n=2, min_genes=2)
    assert set(out) == {"pA"}                                # ZZZ not a GWPS perturbation
    assert out["pA"] == ["C", "B"]                           # |−4| > |3| > rest; target kept for masks


def test_gwps_downstream_sets_min_genes_filter():
    z = pd.DataFrame([[2.0, 1.0]], index=["A"], columns=["A", "B"])
    assert nw.gwps_downstream_sets(z, {"pA": "A"}, ["A", "B"], top_n=5, min_genes=5) == {}
