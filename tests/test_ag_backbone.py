"""Tests for AlphaGenome backbone helpers that need no torch / weights.

Track selection (uses the shipped metadata parquet) and gene-body geometry are
pure and can be exercised without the ``[alphagenome]`` extra beyond pandas.
"""

import os

import numpy as np
import pytest

from concept_shift import ag_backbone

META = os.path.join(os.path.dirname(__file__), "..", "metadata", "track_metadata.parquet")


@pytest.mark.skipif(not os.path.exists(META), reason="track_metadata.parquet missing")
@pytest.mark.parametrize("output_type", ["cage", "rna_seq"])
def test_k562_track_indices(output_type):
    bundle, desc = ag_backbone.k562_track_indices(META, output_type)
    assert set(bundle) == {"all", "plus", "minus"}
    assert len(bundle["all"]) > 0
    # Indices must be valid positions within the head's track count.
    assert max(bundle["all"]) < ag_backbone.AG_NUM_TRACKS[output_type]
    assert "K562" in desc


def test_genebody_indices_plus_strand():
    rf = 1000
    tss = 5000
    # gene body 5000..5100 on + strand -> centred window starts at tss-rf/2.
    idx = ag_backbone._genebody_indices(tss, "+", 5000, 5100, rf)
    assert idx.min() == rf // 2            # tss maps to window centre
    assert idx.max() == rf // 2 + 99
    assert (idx >= 0).all() and (idx < rf).all()


def test_genebody_indices_minus_strand_flips():
    rf = 1000
    idx_plus = ag_backbone._genebody_indices(5000, "+", 5000, 5100, rf)
    idx_minus = ag_backbone._genebody_indices(5000, "-", 5000, 5100, rf)
    # Minus strand positions are the flipped complement within [0, rf).
    assert (idx_minus >= 0).all() and (idx_minus < rf).all()
    assert len(idx_minus) == len(idx_plus)


def test_genebody_indices_out_of_window_empty():
    rf = 1000
    # Gene body far outside the +/- rf/2 window around the TSS.
    idx = ag_backbone._genebody_indices(5000, "+", 50_000, 50_100, rf)
    assert idx.size == 0
